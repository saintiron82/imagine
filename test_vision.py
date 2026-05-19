"""
Vision Module Test Script

Tests the VisionAnalyzer with a sample image.
"""

from pathlib import Path
from backend.vision.analyzer import VisionAnalyzer
from PIL import Image
import sys

def test_vision_analyzer():
    """Test basic vision analysis functionality."""

    print("=== Vision Analyzer Test ===\n")

    # Initialize analyzer
    print("1. Initializing VisionAnalyzer...")
    analyzer = VisionAnalyzer()
    print(f"   Device: {analyzer.device}")
    print(f"   Status: OK\n")

    # Find a test image
    print("2. Looking for test image...")

    # Try to find an existing thumbnail
    thumbnail_dir = Path("output/thumbnails")
    if thumbnail_dir.exists():
        thumbnails = list(thumbnail_dir.glob("*.png"))
        if thumbnails:
            test_image = thumbnails[0]
            print(f"   Found: {test_image.name}\n")
        else:
            print("   No thumbnails found. Please process an image first.")
            return
    else:
        print("   No thumbnails directory. Please process an image first.")
        return

    # Load image
    print("3. Loading image...")
    try:
        image = Image.open(test_image).convert("RGB")
        print(f"   Size: {image.size}")
        print(f"   Mode: {image.mode}\n")
    except Exception as e:
        print(f"   Error loading image: {e}")
        return

    # Analyze image
    print("4. Analyzing image with spatial 2-stage pipeline...")
    print("   This may take 1-2 minutes for model download...")

    try:
        result = analyzer.classify_and_analyze(
            image,
            context={"file_name": test_image.name},
        )

        print("\n=== Analysis Results ===\n")
        tags = result.get("tags") or []
        print(f"Caption: {result.get('caption', '')}")
        print(f"\nTags ({len(tags)}): {', '.join(tags[:10])}")
        print(f"\nOCR Text: {result.get('ocr') or result.get('text_content') or '(none)'}")
        print(f"\nImage Type: {result.get('image_type') or '(none)'}")
        print(f"\nSpatial Objects: {len(result.get('objects') or [])}")

        print("\n=== Test PASSED ===")

    except Exception as e:
        print(f"\n   Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n=== Test FAILED ===")

if __name__ == "__main__":
    test_vision_analyzer()
