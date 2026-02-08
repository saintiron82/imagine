"""Test script for ImageParser."""
from pathlib import Path
from PIL import Image

# Create a simple test image
test_dir = Path("test_assets")
test_dir.mkdir(exist_ok=True)

# Create a simple PNG
img = Image.new('RGB', (100, 100), color=(255, 0, 0))
test_png = test_dir / "test_sample.png"
img.save(test_png)
print(f"Created test image: {test_png}")

# Test the parser
from backend.parser.image_parser import ImageParser

parser = ImageParser(output_dir=Path("output"))
result = parser.parse(test_png)

if result.success:
    print("ImageParser OK!")
    print(f"  Format: {result.asset_meta.format}")
    print(f"  Resolution: {result.asset_meta.resolution}")
    print(f"  Semantic Tags: {result.asset_meta.semantic_tags}")
    print(f"  Thumbnail: {result.asset_meta.thumbnail_url}")
    
    # Check JSON output
    json_path = Path("output/json/test_sample.json")
    if json_path.exists():
        print(f"  JSON saved: {json_path}")
    else:
        print("  ERROR: JSON not saved!")
else:
    print(f"FAILED: {result.errors}")
