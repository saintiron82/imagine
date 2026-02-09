"""
Auto Batch Size Calibrator

Automatically determines optimal batch size for the current system by:
1. Logarithmic search: 1, 2, 4, 8, 16, 32, 64...
2. Detecting efficiency plateaus and system limits
3. Applying 80% safety margin for production use
"""

import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess

logger = logging.getLogger(__name__)


class AutoBatchCalibrator:
    """
    Automatically calibrates optimal batch size for image processing.

    Usage:
        calibrator = AutoBatchCalibrator(tier='standard')
        config = calibrator.calibrate(test_files)
        print(f"Optimal batch size: {config['recommended']}")
    """

    def __init__(
        self,
        tier: str = 'standard',
        max_batch_size: int = 128,
        efficiency_threshold: float = 0.1,  # 10% improvement required
        safety_margin: float = 0.8,  # 80% of optimal
        timeout_per_file: int = 300  # 5 minutes per file max (includes model loading)
    ):
        """
        Initialize calibrator.

        Args:
            tier: AI tier (standard/pro/ultra)
            max_batch_size: Maximum batch size to test
            efficiency_threshold: Minimum efficiency gain to continue (0.1 = 10%)
            safety_margin: Safety factor for recommended batch size (0.8 = 80%)
            timeout_per_file: Timeout in seconds per file
        """
        self.tier = tier
        self.max_batch_size = max_batch_size
        self.efficiency_threshold = efficiency_threshold
        self.safety_margin = safety_margin
        self.timeout_per_file = timeout_per_file

        # Logarithmic batch sizes: 1, 2, 4, 8, 16, 32, 64, 128
        self.batch_sizes = [2**i for i in range(8) if 2**i <= max_batch_size]
        if 1 not in self.batch_sizes:
            self.batch_sizes.insert(0, 1)

    def calibrate(
        self,
        test_files: List[str],
        save_results: bool = True
    ) -> Dict:
        """
        Run calibration to find optimal batch size.

        Args:
            test_files: List of test file paths (should have at least max_batch_size files)
            save_results: Save calibration results to file

        Returns:
            dict: Calibration results
                {
                    'tier': str,
                    'optimal': int,           # Best batch size found
                    'recommended': int,       # Optimal * safety_margin
                    'max_tested': int,        # Largest batch size tested
                    'failed_at': int or None, # Batch size that caused failure
                    'results': {batch_size: time_per_file},
                    'speedup': {batch_size: speedup_vs_single},
                    'timestamp': str
                }
        """
        logger.info(f"[AUTO CALIBRATION] Starting for tier: {self.tier}")
        logger.info(f"[AUTO CALIBRATION] Test batch sizes: {self.batch_sizes}")

        results = {}
        speedups = {}
        baseline_time = None
        optimal_batch = 1
        failed_at = None

        for batch_size in self.batch_sizes:
            # Check if we have enough test files
            if len(test_files) < batch_size:
                logger.warning(f"[AUTO] Not enough test files for batch size {batch_size}, stopping")
                break

            logger.info(f"\n[AUTO] Testing batch size: {batch_size}")

            try:
                # Run benchmark
                time_per_file = self._benchmark_batch(
                    test_files[:batch_size],
                    batch_size
                )

                if time_per_file is None:
                    logger.error(f"[AUTO] Benchmark failed for batch size {batch_size}")
                    failed_at = batch_size
                    break

                results[batch_size] = time_per_file

                # Calculate speedup vs single file
                if baseline_time is None:
                    baseline_time = time_per_file
                    speedups[batch_size] = 1.0
                else:
                    speedups[batch_size] = baseline_time / time_per_file

                logger.info(f"[AUTO] Batch {batch_size}: {time_per_file:.2f}s/file (speedup: {speedups[batch_size]:.1f}x)")

                # Check efficiency gain
                if batch_size > 1:
                    prev_batch = batch_size // 2
                    if prev_batch in speedups:
                        prev_speedup = speedups[prev_batch]
                        curr_speedup = speedups[batch_size]
                        improvement = (curr_speedup - prev_speedup) / prev_speedup

                        logger.info(f"[AUTO] Efficiency improvement: {improvement*100:.1f}%")

                        if improvement < self.efficiency_threshold:
                            logger.info(f"[AUTO] Efficiency plateau reached (< {self.efficiency_threshold*100}% gain)")
                            optimal_batch = prev_batch
                            break

                # Update optimal batch
                optimal_batch = batch_size

            except Exception as e:
                logger.error(f"[AUTO] Failed at batch size {batch_size}: {e}")
                failed_at = batch_size
                break

        # Apply safety margin
        recommended_batch = max(1, int(optimal_batch * self.safety_margin))

        calibration_result = {
            'tier': self.tier,
            'optimal': optimal_batch,
            'recommended': recommended_batch,
            'max_tested': max(results.keys()) if results else 1,
            'failed_at': failed_at,
            'results': results,
            'speedup': speedups,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Log summary
        logger.info(f"\n[AUTO CALIBRATION] Results Summary:")
        logger.info(f"  Optimal batch size: {optimal_batch}")
        logger.info(f"  Recommended (80% safety): {recommended_batch}")
        if speedups:
            logger.info(f"  Max speedup: {max(speedups.values()):.1f}x")
        if failed_at:
            logger.info(f"  System limit: {failed_at} files (VRAM/Memory)")

        # Save results
        if save_results:
            self._save_results(calibration_result)

        return calibration_result

    def _benchmark_batch(
        self,
        files: List[str],
        batch_size: int
    ) -> Optional[float]:
        """
        Benchmark a specific batch size.

        Args:
            files: List of file paths to process
            batch_size: Number of files to process in parallel

        Returns:
            float: Average time per file in seconds, or None if failed
        """
        import json

        files_json = json.dumps(files)

        cmd = [
            "python",
            "backend/pipeline/ingest_engine.py",
            "--files", files_json,
            "--batch-size", str(batch_size)
        ]

        timeout = self.timeout_per_file * batch_size

        try:
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent.parent,
                encoding='utf-8',
                errors='replace'
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"[AUTO] Process failed with exit code {result.returncode}")
                return None

            # Calculate per-file time
            time_per_file = elapsed / batch_size

            return time_per_file

        except subprocess.TimeoutExpired:
            logger.error(f"[AUTO] Timeout ({timeout}s) exceeded")
            return None
        except Exception as e:
            logger.error(f"[AUTO] Benchmark error: {e}")
            return None

    def _save_results(self, results: Dict):
        """Save calibration results to file."""
        output_dir = Path("calibration_results")
        output_dir.mkdir(exist_ok=True)

        filename = f"auto_batch_{self.tier}_{results['timestamp'].replace(':', '-').replace(' ', '_')}.json"
        output_file = output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Also save as "latest"
        latest_file = output_dir / f"auto_batch_{self.tier}_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"[AUTO] Results saved to: {output_file}")

    @staticmethod
    def load_latest_calibration(tier: str) -> Optional[Dict]:
        """
        Load the most recent calibration results for a tier.

        Args:
            tier: AI tier (standard/pro/ultra)

        Returns:
            dict: Calibration results, or None if not found
        """
        latest_file = Path(f"calibration_results/auto_batch_{tier}_latest.json")

        if not latest_file.exists():
            return None

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None


def run_calibration(tier: str, test_files: List[str]) -> Dict:
    """
    Convenience function to run calibration.

    Args:
        tier: AI tier (standard/pro/ultra)
        test_files: List of test file paths

    Returns:
        dict: Calibration results
    """
    calibrator = AutoBatchCalibrator(tier=tier)
    return calibrator.calibrate(test_files)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_batch_calibrator.py <tier> [test_files...]")
        print("Example: python auto_batch_calibrator.py standard test_assets/*.psd")
        sys.exit(1)

    tier = sys.argv[1]
    test_files = sys.argv[2:] if len(sys.argv) > 2 else []

    # Auto-discover test files if not provided
    if not test_files:
        test_dir = Path("test_assets")
        test_files = [str(f) for f in test_dir.glob("*.psd")] + \
                    [str(f) for f in test_dir.glob("*.png")]
        test_files = test_files[:64]  # Max 64 for calibration

    print(f"Running AUTO calibration for tier: {tier}")
    print(f"Test files: {len(test_files)}")

    config = run_calibration(tier, test_files)

    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Optimal batch size: {config['optimal']}")
    print(f"Recommended (safe): {config['recommended']}")
    print(f"Max speedup: {max(config['speedup'].values()):.1f}x")
