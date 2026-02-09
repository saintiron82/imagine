"""
Test Ollama Parallel Request Handling

Tests whether Ollama processes multiple requests:
1. Sequentially (queueing)
2. In parallel (concurrent)
"""

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3-vl:8b"

def send_request(request_id: int, prompt: str):
    """Send a single request to Ollama and measure time."""
    url = f"{OLLAMA_HOST}/api/generate"

    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} starting...")
    start = time.time()

    try:
        response = requests.post(url, json=data, timeout=180)
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} DONE in {elapsed:.1f}s")
            return {
                'id': request_id,
                'success': True,
                'time': elapsed,
                'response_length': len(result.get('response', ''))
            }
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} FAILED: {response.status_code}")
            return {
                'id': request_id,
                'success': False,
                'time': elapsed,
                'error': f"Status {response.status_code}"
            }
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} ERROR in {elapsed:.1f}s: {e}")
        return {
            'id': request_id,
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def test_sequential():
    """Test sequential processing (baseline)."""
    print("\n" + "="*70)
    print("TEST 1: SEQUENTIAL (Baseline)")
    print("="*70)

    prompts = [
        "Describe a sunset",
        "Describe a forest",
        "Describe a mountain"
    ]

    start = time.time()
    results = []

    for i, prompt in enumerate(prompts, 1):
        result = send_request(i, prompt)
        results.append(result)

    total = time.time() - start

    print(f"\n{'='*70}")
    print(f"Sequential Results:")
    print(f"  Total time: {total:.1f}s")
    print(f"  Avg per request: {total/len(prompts):.1f}s")
    print(f"{'='*70}\n")

    return results, total


def test_parallel(num_workers=3):
    """Test parallel processing."""
    print("\n" + "="*70)
    print(f"TEST 2: PARALLEL ({num_workers} concurrent requests)")
    print("="*70)

    prompts = [
        "Describe a sunset",
        "Describe a forest",
        "Describe a mountain"
    ]

    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request, i, prompt)
            for i, prompt in enumerate(prompts, 1)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total = time.time() - start

    print(f"\n{'='*70}")
    print(f"Parallel Results:")
    print(f"  Total time: {total:.1f}s")
    print(f"  Avg per request: {total/len(prompts):.1f}s")
    print(f"{'='*70}\n")

    return results, total


def main():
    print("="*70)
    print("OLLAMA PARALLEL REQUEST TEST")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Host: {OLLAMA_HOST}")

    # Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n⚠️  WARNING: Ollama server may not be running")
            print(f"Status code: {response.status_code}")
            return
    except Exception as e:
        print("\n❌ ERROR: Cannot connect to Ollama server")
        print(f"Error: {e}")
        print(f"\nMake sure Ollama is running: ollama serve")
        return

    # Test 1: Sequential
    seq_results, seq_time = test_sequential()

    # Wait between tests
    print("\nWaiting 5 seconds before parallel test...\n")
    time.sleep(5)

    # Test 2: Parallel
    par_results, par_time = test_parallel(num_workers=3)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print(f"\nSequential: {seq_time:.1f}s total ({seq_time/3:.1f}s avg)")
    print(f"Parallel:   {par_time:.1f}s total ({par_time/3:.1f}s avg)")

    speedup = seq_time / par_time
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 2.5:
        print("\n✅ RESULT: Ollama processes requests IN PARALLEL!")
        print("   Multiple requests are handled concurrently.")
        print("   ⚡ Parallel Ollama adapter will work!")
    elif speedup > 1.5:
        print("\n⚠️  RESULT: Partial parallelization")
        print("   Some concurrency but not full parallel processing.")
        print("   Consider vLLM for better performance.")
    else:
        print("\n❌ RESULT: Ollama processes requests SEQUENTIALLY")
        print("   Requests are queued and processed one at a time.")
        print("   💡 Recommendation: Use vLLM for parallel processing")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
