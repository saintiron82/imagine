"""
Test Vision Backends - Compare Transformers vs Ollama.

Usage:
    python tools/test_vision_backends.py --image test.png
    python tools/test_vision_backends.py --image test.png --backend ollama
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image


def test_backend(image_path: str, backend: str):
    """Test specific vision backend."""
    print(f"\n{'='*80}")
    print(f"Testing {backend.upper()} backend")
    print(f"{'='*80}\n")

    # Set environment
    os.environ['VISION_BACKEND'] = backend

    # Import factory (after setting env)
    from backend.vision.vision_factory import get_vision_analyzer

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"[IMAGE] Loaded: {image_path}")
    print(f"   Size: {image.size[0]}x{image.size[1]} pixels\n")

    # Analyze
    print(f"[ANALYZE] Running {backend} analysis...")
    start = time.time()

    analyzer = get_vision_analyzer()
    result = analyzer.analyze(image)

    duration = time.time() - start

    # Display results
    print(f"\n[OK] Analysis complete in {duration:.2f}s\n")
    print(f"Caption: {result.get('caption', 'N/A')}")
    print(f"\nTags: {', '.join(result.get('tags', [])[:10])}")

    if result.get('ocr'):
        print(f"\nOCR Text: {result.get('ocr')[:100]}...")

    print(f"\nDominant Color: {result.get('color', 'N/A')}")
    print(f"Style: {result.get('style', 'N/A')}")

    return result, duration


def compare_backends(image_path: str):
    """Compare both backends."""
    print(f"\n{'='*80}")
    print(f"COMPARING VISION BACKENDS")
    print(f"{'='*80}")

    results = {}

    # Test Transformers
    try:
        result_tf, time_tf = test_backend(image_path, 'transformers')
        results['transformers'] = {'result': result_tf, 'time': time_tf, 'success': True}
    except Exception as e:
        print(f"❌ Transformers failed: {e}")
        results['transformers'] = {'success': False, 'error': str(e)}

    # Reset factory
    from backend.vision.vision_factory import VisionAnalyzerFactory
    VisionAnalyzerFactory.reset()

    # Test Ollama
    try:
        result_ol, time_ol = test_backend(image_path, 'ollama')
        results['ollama'] = {'result': result_ol, 'time': time_ol, 'success': True}
    except Exception as e:
        print(f"❌ Ollama failed: {e}")
        results['ollama'] = {'success': False, 'error': str(e)}

    # Summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    for backend, data in results.items():
        if data['success']:
            print(f"✅ {backend.upper()}: {data['time']:.2f}s")
            print(f"   Caption length: {len(data['result'].get('caption', ''))}")
            print(f"   Tags count: {len(data['result'].get('tags', []))}")
        else:
            print(f"❌ {backend.upper()}: {data['error']}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test vision backends')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--backend', choices=['transformers', 'ollama', 'both'],
                        default='both', help='Backend to test')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    if args.backend == 'both':
        compare_backends(args.image)
    else:
        test_backend(args.image, args.backend)
