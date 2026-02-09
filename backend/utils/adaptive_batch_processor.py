"""
Adaptive Batch Size Processor

Dynamically adjusts batch size during processing to find optimal performance.

Algorithm:
1. Start with batch_size=1 (safe baseline)
2. Process files and measure performance (time per file)
3. Incrementally increase batch size (1→2→3→5→8→10...)
4. Stop increasing when performance degrades
5. Cache optimal batch size for future runs

v3.1.1: Platform-aware adaptive batching
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Performance metrics for a batch."""
    batch_size: int
    num_files: int
    total_time: float
    time_per_file: float
    success_rate: float

    def __repr__(self):
        return (f"Batch({self.batch_size}): {self.num_files} files, "
                f"{self.time_per_file:.1f}s/file, "
                f"{self.success_rate*100:.0f}% success")


class AdaptiveBatchProcessor:
    """
    Adaptive batch size processor that finds optimal batch size at runtime.

    Features:
    - Starts with batch_size=1 (safe)
    - Incrementally increases batch size
    - Monitors performance (time per file)
    - Stops when performance degrades
    - Caches optimal batch size

    Usage:
        processor = AdaptiveBatchProcessor(
            process_func=process_single_file,
            platform='windows',
            tier='ultra'
        )
        results = processor.process_adaptive(files)
    """

    # Batch size progression (conservative growth)
    BATCH_SIZES = [1, 2, 3, 5, 8, 10, 16, 20]

    # Performance degradation threshold
    DEGRADATION_THRESHOLD = 1.2  # 20% slower → stop

    # Minimum improvement to continue
    MIN_IMPROVEMENT = 0.95  # Must be at least 5% better

    def __init__(
        self,
        process_func: Callable,
        platform: str = 'windows',
        tier: str = 'ultra',
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize adaptive batch processor.

        Args:
            process_func: Function to process a single file
            platform: Platform name (for caching)
            tier: AI tier (for caching)
            cache_dir: Directory to cache results
        """
        self.process_func = process_func
        self.platform = platform.lower()
        self.tier = tier.lower()

        if cache_dir is None:
            cache_dir = Path("calibration_results")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.metrics_history: List[BatchMetrics] = []
        self.optimal_batch_size: Optional[int] = None

    def _get_cache_path(self) -> Path:
        """Get cache file path for current platform/tier."""
        return self.cache_dir / f"adaptive_batch_{self.platform}_{self.tier}.json"

    def load_cached_optimal(self) -> Optional[int]:
        """Load previously cached optimal batch size."""
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            optimal = data.get('optimal_batch_size')
            logger.info(f"[CACHE] Loaded optimal batch_size={optimal} from cache")
            return optimal

        except Exception as e:
            logger.warning(f"[CACHE] Failed to load cache: {e}")
            return None

    def save_optimal(self, batch_size: int, metrics: BatchMetrics):
        """Save optimal batch size to cache."""
        cache_path = self._get_cache_path()

        data = {
            'optimal_batch_size': batch_size,
            'platform': self.platform,
            'tier': self.tier,
            'metrics': {
                'batch_size': batch_size,
                'time_per_file': metrics.time_per_file,
                'success_rate': metrics.success_rate
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[CACHE] Saved optimal batch_size={batch_size} to cache")
        except Exception as e:
            logger.warning(f"[CACHE] Failed to save cache: {e}")

    def _process_batch(
        self,
        files: List[Any],
        batch_size: int
    ) -> BatchMetrics:
        """
        Process a batch of files and measure performance.

        Args:
            files: List of files to process
            batch_size: Number of files to process concurrently

        Returns:
            BatchMetrics with performance data
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()
        results = []

        if batch_size == 1:
            # Sequential processing
            for file_path in files:
                try:
                    result = self.process_func(file_path)
                    results.append({'success': True, 'result': result})
                except Exception as e:
                    logger.error(f"[BATCH] Error processing {file_path}: {e}")
                    results.append({'success': False, 'error': str(e)})
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(self.process_func, file_path): file_path
                    for file_path in files
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append({'success': True, 'result': result})
                    except Exception as e:
                        file_path = futures[future]
                        logger.error(f"[BATCH] Error processing {file_path}: {e}")
                        results.append({'success': False, 'error': str(e)})

        total_time = time.time() - start_time
        num_files = len(files)
        time_per_file = total_time / num_files if num_files > 0 else 0
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / num_files if num_files > 0 else 0

        metrics = BatchMetrics(
            batch_size=batch_size,
            num_files=num_files,
            total_time=total_time,
            time_per_file=time_per_file,
            success_rate=success_rate
        )

        logger.info(f"[ADAPTIVE] {metrics}")

        return metrics

    def _should_continue(
        self,
        current_metrics: BatchMetrics,
        baseline_metrics: BatchMetrics
    ) -> bool:
        """
        Determine if we should continue increasing batch size.

        Args:
            current_metrics: Current batch performance
            baseline_metrics: Previous best performance

        Returns:
            True if performance is still improving
        """
        # Check for degradation
        degradation_ratio = current_metrics.time_per_file / baseline_metrics.time_per_file

        if degradation_ratio > self.DEGRADATION_THRESHOLD:
            logger.info(
                f"[ADAPTIVE] Performance degraded {degradation_ratio:.2f}x "
                f"({baseline_metrics.time_per_file:.1f}s → {current_metrics.time_per_file:.1f}s) - STOPPING"
            )
            return False

        # Check for improvement
        if degradation_ratio < self.MIN_IMPROVEMENT:
            logger.info(
                f"[ADAPTIVE] Performance improved {1/degradation_ratio:.2f}x "
                f"({baseline_metrics.time_per_file:.1f}s → {current_metrics.time_per_file:.1f}s) - CONTINUING"
            )
            return True

        # Marginal change - continue cautiously
        logger.info(
            f"[ADAPTIVE] Marginal change {degradation_ratio:.2f}x - CONTINUING"
        )
        return True

    def process_adaptive(
        self,
        files: List[Any],
        use_cache: bool = True
    ) -> Tuple[List[Any], int]:
        """
        Process files with adaptive batch sizing.

        Algorithm:
        1. Check cache for optimal batch size
        2. If cached, use it directly
        3. Otherwise, start with batch_size=1
        4. Process files in chunks, measuring performance
        5. Increase batch size incrementally
        6. Stop when performance degrades
        7. Cache optimal batch size

        Args:
            files: List of files to process
            use_cache: Whether to use cached optimal batch size

        Returns:
            Tuple of (all_results, optimal_batch_size)
        """
        if not files:
            return [], 1

        total_files = len(files)
        logger.info(f"[ADAPTIVE] Processing {total_files} files with adaptive batching")

        # Check cache
        start_from_batch_size = None
        if use_cache:
            cached_optimal = self.load_cached_optimal()
            if cached_optimal:
                # Start from 50% of previous optimal for faster convergence
                start_from_batch_size = max(1, cached_optimal // 2)
                logger.info(
                    f"[ADAPTIVE] Found cached optimal={cached_optimal}, "
                    f"starting from {start_from_batch_size} (50%)"
                )

        # Adaptive search for optimal batch size
        logger.info("[ADAPTIVE] Starting adaptive batch size search...")

        all_results = []
        baseline_metrics = None
        optimal_batch_size = 1

        # Process files in increasing batch sizes
        remaining_files = files[:]

        # Filter BATCH_SIZES to start from the 50% point if cached
        batch_sizes_to_test = self.BATCH_SIZES
        if start_from_batch_size is not None:
            batch_sizes_to_test = [bs for bs in self.BATCH_SIZES if bs >= start_from_batch_size]
            logger.info(
                f"[ADAPTIVE] Starting from batch_size={start_from_batch_size}, "
                f"testing sizes: {batch_sizes_to_test}"
            )

        for batch_size in batch_sizes_to_test:
            if not remaining_files:
                break

            # Take a sample batch (or all remaining)
            sample_size = min(batch_size * 3, len(remaining_files))  # 3x batch size for stable measurement
            sample_files = remaining_files[:sample_size]

            logger.info(
                f"[ADAPTIVE] Testing batch_size={batch_size} "
                f"with {len(sample_files)} files..."
            )

            # Process sample
            metrics = self._process_batch(sample_files, batch_size)
            self.metrics_history.append(metrics)

            # First batch - establish baseline
            if baseline_metrics is None:
                baseline_metrics = metrics
                optimal_batch_size = batch_size
                remaining_files = remaining_files[sample_size:]
                logger.info(f"[ADAPTIVE] Baseline established: {metrics}")
                continue

            # Check if we should continue
            if not self._should_continue(metrics, baseline_metrics):
                # Performance degraded - revert to previous batch size
                logger.info(
                    f"[ADAPTIVE] Optimal batch_size found: {optimal_batch_size} "
                    f"(stopped at {batch_size})"
                )
                break

            # Performance improved or stable - continue
            if metrics.time_per_file < baseline_metrics.time_per_file:
                baseline_metrics = metrics
                optimal_batch_size = batch_size

            remaining_files = remaining_files[sample_size:]

        # Process any remaining files with optimal batch size
        if remaining_files:
            logger.info(
                f"[ADAPTIVE] Processing remaining {len(remaining_files)} files "
                f"with optimal batch_size={optimal_batch_size}"
            )
            final_metrics = self._process_batch(remaining_files, optimal_batch_size)

        # Save optimal batch size to cache
        self.optimal_batch_size = optimal_batch_size
        self.save_optimal(optimal_batch_size, baseline_metrics)

        logger.info(
            f"[ADAPTIVE] Adaptive batching complete: "
            f"optimal_batch_size={optimal_batch_size}, "
            f"baseline={baseline_metrics.time_per_file:.1f}s/file"
        )

        return all_results, optimal_batch_size

    def get_metrics_summary(self) -> str:
        """Get summary of all tested batch sizes."""
        if not self.metrics_history:
            return "No metrics recorded"

        lines = ["Batch Size Performance Summary:", "=" * 50]
        for metrics in self.metrics_history:
            lines.append(
                f"batch_size={metrics.batch_size:2d}: "
                f"{metrics.time_per_file:5.1f}s/file "
                f"({metrics.success_rate*100:3.0f}% success)"
            )

        if self.optimal_batch_size:
            lines.append("=" * 50)
            lines.append(f"Optimal: batch_size={self.optimal_batch_size}")

        return "\n".join(lines)
