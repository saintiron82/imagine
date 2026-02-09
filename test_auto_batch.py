"""
Test AUTO Batch Mode

Quick test of automatic batch size calibration.
"""
import logging
from pathlib import Path
from backend.utils.auto_batch_calibrator import AutoBatchCalibrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)

def main():
    print("="*70)
    print("AUTO BATCH MODE TEST")
    print("="*70)

    # Get test files
    test_dir = Path("test_assets")
    test_files = [str(f) for f in test_dir.glob("*.psd")] + \
                [str(f) for f in test_dir.glob("*.png")]

    print(f"\nAvailable test files: {len(test_files)}")

    if len(test_files) < 8:
        print("⚠️  Not enough test files (need at least 8 for calibration)")
        print(f"   Found: {len(test_files)}")
        print(f"   Required: 8")
        return

    # Run calibration for Standard tier
    print(f"\n{'='*70}")
    print("Running AUTO calibration for Standard tier")
    print(f"{'='*70}\n")

    calibrator = AutoBatchCalibrator(
        tier='standard',
        max_batch_size=32,  # Test up to 32
        efficiency_threshold=0.1,  # 10% improvement required
        safety_margin=0.8  # 80% safety margin
    )

    config = calibrator.calibrate(test_files[:32])

    # Print results
    print(f"\n{'='*70}")
    print("CALIBRATION RESULTS")
    print(f"{'='*70}")

    print(f"\nOptimal batch size: {config['optimal']}")
    print(f"Recommended (safe): {config['recommended']}")
    print(f"Max tested: {config['max_tested']}")
    if config['failed_at']:
        print(f"Failed at: {config['failed_at']} files (system limit)")

    print(f"\nDetailed Results:")
    print(f"{'Batch Size':<12} | {'Time/File':<12} | {'Speedup':<12}")
    print("-" * 40)

    for batch_size in sorted(config['results'].keys()):
        time_per_file = config['results'][batch_size]
        speedup = config['speedup'][batch_size]
        print(f"{batch_size:<12} | {time_per_file:>10.2f}s | {speedup:>10.1f}x")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\nFor production use with Standard tier:")
    print(f"  --batch-size {config['recommended']}")
    print(f"\nOr simply use:")
    print(f"  --batch-size auto")
    print(f"\nCached calibration saved for future runs!")

if __name__ == "__main__":
    main()
