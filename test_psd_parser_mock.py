"""
Mock test for PSDParser.
Since we don't have a real PSD file in the CI/Dev environment,
we mock the psd_tools library to verify the parser logic.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock psd_tools before importing PSDParser if needed, 
# but here we'll patch it during the test.
from backend.parser.psd_parser import PSDParser
from backend.parser.schema import AssetMeta

def test_psd_parser_logic():
    print("Testing PSDParser with Mocks...")
    
    # Create a mock PSD object
    mock_psd = MagicMock()
    mock_psd.width = 1000
    mock_psd.height = 800
    mock_psd.color_mode = 'RGB'
    mock_psd.channels = 3
    
    # Mock descandants (count)
    mock_psd.descendants.return_value = range(5)  # 5 descendant layers
    
    # Mock composite image (PIL Image)
    mock_composite = MagicMock()
    mock_composite.size = (1000, 800)
    mock_composite.mode = 'RGBA'
    mock_psd.composite.return_value = mock_composite
    
    # Mock Layers
    # Layer 1: Background
    l1 = MagicMock()
    l1.name = "Background"
    l1.kind = "pixel"
    l1.visible = True
    l1.opacity = 255
    l1.left, l1.top = 0, 0
    l1.width, l1.height = 1000, 800
    l1.__iter__ = None # Not a group
    
    # Layer 2: Character Group
    l2 = MagicMock()
    l2.name = "Character Group" 
    l2.kind = "group"
    l2.visible = True
    l2.opacity = 255
    l2.left, l2.top = 200, 200
    l2.width, l2.height = 400, 600
    
    # Layer 3: Body (inside Character)
    l3 = MagicMock()
    l3.name = "Body"
    l3.kind = "pixel"
    l3.visible = True
    l3.opacity = 255
    l3.left, l3.top = 200, 200
    l3.width, l3.height = 400, 600
    l3.__iter__ = None
    
    # Layer 4: Text Layer
    l4 = MagicMock()
    l4.name = "Dialogue"
    l4.kind = "type"
    l4.visible = True
    l4.opacity = 255
    l4.left, l4.top = 300, 100
    l4.width, l4.height = 200, 50
    l4.text = "Hello World" # Fallback text
    l4.engine_dict = {
        'Editor': {'Text': 'Hello World'},
        'ResourceDict': {'FontSet': [{'Name': 'Arial-Bold'}]}
    }
    l4.__iter__ = None
    
    # Setup hierarchy
    # Root -> [l1, l2, l4]
    # l2 -> [l3]
    mock_psd.__iter__.return_value = iter([l1, l2, l4])
    l2.__iter__.return_value = iter([l3])
    
    # Patch PSDImage.open to return our mock
    with patch('psd_tools.PSDImage.open', return_value=mock_psd) as mock_open:
        # Initialize Parser
        parser = PSDParser(output_dir=Path("output_mock"))
        
        # Create dummy file to pass validation
        mock_dir = Path("mock_assets")
        mock_dir.mkdir(exist_ok=True)
        fake_path = mock_dir / "test.psd"
        fake_path.touch()
        
        try:
            # Test Parse
            result = parser.parse(fake_path)
            
            if result.success:
                print("✅ PSDParser Logic: SUCCESS")
                meta = result.asset_meta
                
                # Verify File Info
                print(f"  Format: {meta.format}")
                assert meta.format == 'PSD'
                
                # Verify Layer Count
                print(f"  Layer Count: {meta.layer_count}")
                
                # Verify Semantic Tags
                print(f"  Semantic Tags: {meta.semantic_tags}")
                # Expected: "Background Character Group Body Dialogue" (normalized)
                assert "Background" in meta.semantic_tags
                assert "Body" in meta.semantic_tags
                
                # Verify Text Content
                print(f"  Text Content: {meta.text_content}")
                assert "Hello World" in meta.text_content
                
                # Verify Fonts
                print(f"  Fonts: {meta.used_fonts}")
                assert "Arial-Bold" in meta.used_fonts
                
                # Verify Thumbnail
                print(f"  Thumbnail: {meta.thumbnail_url}")
                
                print("ALL CHECKS PASSED")
                
            else:
                print(f"❌ FAILED: {result.errors}")
                exit(1)
        finally:
            # Cleanup
            if fake_path.exists():
                fake_path.unlink()
            if mock_dir.exists():
                mock_dir.rmdir()

if __name__ == "__main__":
    test_psd_parser_logic()
