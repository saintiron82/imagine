"""Batch Processing Performance Benchmark - All English version"""
import subprocess
import time
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime

TEST_DIR = Path(r"C:\Users\saint\ImageParser\test_assets")
CONFIG_FILE = Path(r"C:\Users\saint\ImageParser\config.yaml")
BACKUP_CONFIG = Path(r"C:\Users\saint\ImageParser\config.yaml.backup")
TIERS = ["standard", "pro", "ultra"]
BATCH_SIZES = [1, 5, 10, 20]

results = {
    "timestamp": datetime.now().isoformat(),
    "tiers": {}
}

def backup_config():
    shutil.copy2(CONFIG_FILE, BACKUP_CONFIG)
    print(f"[OK] Config backed up")

def restore_config():
    shutil.copy2(BACKUP_CONFIG, CONFIG_FILE)
    print(f"[OK] Config restored")

def set_tier(tier_name):
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['ai_mode']['override'] = tier_name
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"[OK] Tier set to: {tier_name}")

def get_test_files(count):
    all_files = list(TEST_DIR.glob("*.png")) + list(TEST_DIR.glob("*.psd"))
    return [str(f) for f in all_files[:count]]

def run_test(tier, batch_size):
    print(f"\n{'='*60}")
    print(f"Tier: {tier.upper()} | Batch: {batch_size} files")
    print(f"{'='*60}")

    files = get_test_files(batch_size)
    if len(files) < batch_size:
        print(f"[WARN] Not enough files: {len(files)}/{batch_size}")
        return None

    files_json = json.dumps(files)
    cmd = [
        "python",
        "backend/pipeline/ingest_engine.py",
        "--files", files_json,
        "--batch-size", str(batch_size)
    ]

    print(f"Processing {batch_size} files...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=r"C:\Users\saint\ImageParser",
            encoding='utf-8',
            errors='replace'
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            per_file = elapsed / batch_size
            print(f"[OK] Done: {elapsed:.1f}s (avg {per_file:.1f}s/file)")
            return {
                'status': 'success',
                'total_time': round(elapsed, 2),
                'per_file': round(per_file, 2),
                'files_count': batch_size
            }
        else:
            print(f"[FAIL] Error occurred")
            return {'status': 'failed'}

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded 10 minutes")
        return {'status': 'timeout'}
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {'status': 'error'}

def main():
    print("="*70)
    print("BATCH PROCESSING PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Test files: {TEST_DIR}")
    print(f"Tiers: {', '.join(TIERS)}")
    print(f"Batch sizes: {', '.join(map(str, BATCH_SIZES))}")
    print(f"Estimated time: ~20 minutes\n")

    backup_config()

    try:
        for tier in TIERS:
            print(f"\n\n{'#'*70}")
            print(f"# TIER: {tier.upper()}")
            print(f"{'#'*70}")

            set_tier(tier)
            time.sleep(1)

            results['tiers'][tier] = {}

            for batch_size in BATCH_SIZES:
                result = run_test(tier, batch_size)
                if result:
                    results['tiers'][tier][batch_size] = result
                time.sleep(2)

        output_file = Path("benchmark_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n\n{'='*70}")
        print(f"[OK] BENCHMARK COMPLETED!")
        print(f"Results: {output_file}")
        print(f"{'='*70}")

        print_results(results)

    finally:
        restore_config()

def print_results(results):
    print("\n\n" + "="*80)
    print("BATCH PROCESSING PERFORMANCE RESULTS")
    print("="*80)

    print("\nTOTAL TIME (seconds)")
    print("-"*80)
    print(f"{'Tier':<12} | {'1 file':<10} | {'5 files':<10} | {'10 files':<10} | {'20 files':<10}")
    print("-"*80)

    for tier in TIERS:
        tier_data = results['tiers'].get(tier, {})
        row = f"{tier.upper():<12} |"
        for batch_size in BATCH_SIZES:
            data = tier_data.get(batch_size, {})
            if data.get('status') == 'success':
                row += f" {data['total_time']:>8.1f}s |"
            else:
                row += f" {'FAIL':>8} |"
        print(row)

    print("\nAVERAGE TIME (seconds/file)")
    print("-"*80)
    print(f"{'Tier':<12} | {'1 file':<10} | {'5 files':<10} | {'10 files':<10} | {'20 files':<10}")
    print("-"*80)

    for tier in TIERS:
        tier_data = results['tiers'].get(tier, {})
        row = f"{tier.upper():<12} |"
        for batch_size in BATCH_SIZES:
            data = tier_data.get(batch_size, {})
            if data.get('status') == 'success':
                row += f" {data['per_file']:>8.1f}s |"
            else:
                row += f" {'-':>8} |"
        print(row)

    print("\nBATCH EFFICIENCY ANALYSIS")
    print("-"*80)
    for tier in TIERS:
        tier_data = results['tiers'].get(tier, {})
        single = tier_data.get(1, {}).get('per_file')
        batch_20 = tier_data.get(20, {}).get('per_file')

        if single and batch_20:
            speedup = single / batch_20
            savings = (1 - batch_20/single) * 100
            print(f"{tier.upper():<12}: {single:.1f}s -> {batch_20:.1f}s "
                  f"(speedup: {speedup:.1f}x, saved: {savings:.0f}%)")

if __name__ == "__main__":
    main()
