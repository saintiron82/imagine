"""E2E Integration Test with externally supplied real assets."""

import os
from pathlib import Path

import pytest

OUTPUT_DIR = Path("output")

def test_e2e_real_psd():
    print("🚀 Starting E2E Test with Real PSDs...")

    assets_dir_raw = os.environ.get("IMAGINE_E2E_ASSETS_DIR")
    if not assets_dir_raw:
        pytest.skip("Set IMAGINE_E2E_ASSETS_DIR to run real-asset E2E tests")

    test_assets_dir = Path(assets_dir_raw).expanduser()
    if not test_assets_dir.exists():
        pytest.skip(f"Real-asset directory does not exist: {test_assets_dir}")

    # Get all PSD files
    psd_files = list(test_assets_dir.glob("*.psd"))
    if not psd_files:
        pytest.skip(f"No PSD files found in {test_assets_dir}")

    from backend.pipeline.ingest_engine import process_file
    from backend.parser.schema import AssetMeta

    OUTPUT_DIR.mkdir(exist_ok=True)
        
    print(f"Found {len(psd_files)} PSDs: {[f.name for f in psd_files]}")
    
    for psd_file in psd_files:
        print(f"\nProcessing {psd_file.name}...")
        process_file(psd_file)
        
        # Verify JSON Output
        json_path = OUTPUT_DIR / "json" / f"{psd_file.stem}.json"
        if not json_path.exists():
            print(f"❌ JSON output missing for {psd_file.name}")
            exit(1)
            
        # Verify Thumbnail Output
        thumb_path = OUTPUT_DIR / "thumbnails" / f"{psd_file.stem}_thumb.png"
        if not thumb_path.exists():
            print(f"❌ Thumbnail output missing for {psd_file.name}")
            exit(1)
            
        # Validate JSON Content
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                asset_meta = AssetMeta.model_validate_json(json_content)
                
            print(f"✅ {psd_file.name} Parsed Successfully!")
            print(f"   - Layers: {asset_meta.layer_count}")
            print(f"   - Tags: {asset_meta.semantic_tags}")
            print(f"   - Fonts: {len(asset_meta.used_fonts)}")
            
            # Additional checks
            if asset_meta.format != 'PSD':
                print(f"❌ Format mismatch: {asset_meta.format}")
                exit(1)
                
        except Exception as e:
            print(f"❌ JSON Validation Failed for {psd_file.name}: {e}")
            exit(1)

    print("\n✨ All Real PSD Tests Passed! Phase 1 Complete. ✨")

if __name__ == "__main__":
    test_e2e_real_psd()
