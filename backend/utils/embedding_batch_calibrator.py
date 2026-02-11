"""
Embedding Batch Calibrator - Auto-discover optimal model batch size.

DEPRECATED (v3.3): Superseded by adaptive batch sizing built into
SigLIP2Encoder.encode_image_batch() which handles probe → scale-up →
fine-tune → OOM recovery dynamically at runtime.

Kept for reference only. No longer imported by ingest_engine.

v3.2: Integrates with Phase 3 of the batch pipeline.
"""

import gc
import json
import logging
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "calibration_results"

# OOM error patterns (CUDA + MPS + generic)
OOM_PATTERNS = [
    "out of memory",
    "cuda out of memory",
    "mps backend",
    "mps out of memory",
    "allocation",
    "allocate",
]


@dataclass
class ProbeResult:
    batch_size: int
    time_per_image: float
    success: bool
    total_time: float = 0.0


class EmbeddingBatchCalibrator:
    """
    GPU model batch size auto-calibrator.

    Probes increasing batch sizes with real images, measures throughput,
    stops at OOM or performance degradation. Caches results for reuse.
    """

    VV_BATCH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

    DEGRADATION_THRESHOLD = 1.15   # 15% slower → stop
    SAFETY_MARGIN = 0.85           # Use 85% of optimal
    MIN_PROBE_IMAGES = 4           # Need at least 4 images for calibration

    CACHE_VERSION = 1

    def __init__(self, model_name: str, device: str):
        from ..utils.tier_config import get_active_tier
        tier_name, tier_config = get_active_tier()

        self._model_name = model_name
        self._device = device
        self._tier = tier_name
        self._max_edge = tier_config.get("preprocess", {}).get("max_edge", 768)

    @staticmethod
    def _is_oom(error: Exception) -> bool:
        msg = str(error).lower()
        return any(p in msg for p in OOM_PATTERNS)

    def _cache_path(self) -> Path:
        os_name = platform.system().lower()
        model_short = self._model_name.split("/")[-1][:30].replace(" ", "-").lower()
        name = f"embed_batch_vv_{os_name}_{self._tier}_{model_short}_{self._device}.json"
        return CACHE_DIR / name

    def _load_cache(self) -> Optional[dict]:
        """Load cached calibration data if valid. Returns full dict or None."""
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if (
                data.get("cache_version") == self.CACHE_VERSION
                and data.get("model") == self._model_name
                and data.get("device") == self._device
                and data.get("tier") == self._tier
                and data.get("max_edge") == self._max_edge
            ):
                return data
            logger.info("[CALIBRATE] Cache invalidated (config changed)")
        except Exception:
            pass
        return None

    def _save_cache(self, recommended: int, optimal: int, probes: List[ProbeResult]):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "recommended_batch_size": recommended,
            "optimal_batch_size": optimal,
            "model": self._model_name,
            "device": self._device,
            "tier": self._tier,
            "max_edge": self._max_edge,
            "platform": platform.system().lower(),
            "safety_margin": self.SAFETY_MARGIN,
            "cache_version": self.CACHE_VERSION,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "probe_results": [asdict(p) for p in probes],
        }
        path = self._cache_path()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"[CALIBRATE] Saved cache: {path.name}")

    def _probe_batch_size(self, encoder, images, batch_size: int) -> ProbeResult:
        """Test a single batch size. Returns ProbeResult."""
        import torch

        subset = images[:batch_size]
        actual_count = len(subset)

        t0 = time.perf_counter()
        try:
            vectors = encoder.encode_image_batch(subset, batch_size=batch_size)
            elapsed = time.perf_counter() - t0
            return ProbeResult(
                batch_size=batch_size,
                time_per_image=elapsed / max(actual_count, 1),
                success=True,
                total_time=elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            if self._is_oom(e):
                logger.warning(f"[CALIBRATE] OOM at batch_size={batch_size}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                return ProbeResult(
                    batch_size=batch_size,
                    time_per_image=float("inf"),
                    success=False,
                    total_time=elapsed,
                )
            raise

    def calibrate_vv(
        self, encoder, probe_images: list
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Auto-discover optimal VV batch size.

        Args:
            encoder: SigLIP2Encoder instance (must be loaded)
            probe_images: List of PIL Images to use as probes

        Returns:
            (recommended_batch_size, probe_vectors)
            probe_vectors: Vectors from the final successful probe (for reuse)
        """
        # CPU → skip calibration
        if self._device == "cpu":
            logger.info("[CALIBRATE] CPU mode, using default batch_size=4")
            return 4, []

        # Too few images → skip calibration
        if len(probe_images) < self.MIN_PROBE_IMAGES:
            logger.info(
                f"[CALIBRATE] Too few images ({len(probe_images)}) "
                f"for calibration, using default=4"
            )
            return 4, []

        # Check cache → use as starting hint (probe from 50% of cached optimal)
        cached_data = self._load_cache()
        probe_sizes = list(self.VV_BATCH_SIZES)

        if cached_data is not None:
            cached_optimal = cached_data.get("optimal_batch_size", 8)
            start_from = max(1, cached_optimal // 2)
            # Filter: probe only from 50% of cached optimal upward
            probe_sizes = [bs for bs in probe_sizes if bs >= start_from]
            if not probe_sizes:
                probe_sizes = [self.VV_BATCH_SIZES[-1]]
            logger.info(
                f"[CALIBRATE] Cache hint: optimal was {cached_optimal}, "
                f"probing from {probe_sizes[0]}"
            )
        else:
            logger.info("[CALIBRATE] No cache, full probe")

        logger.info(
            f"[CALIBRATE] Probing VV batch sizes {probe_sizes} "
            f"(model={self._model_name}, device={self._device})..."
        )

        # Ensure model is loaded before probing
        encoder._load()

        probes: List[ProbeResult] = []
        best_tpi = float("inf")
        optimal_bs = 1
        last_successful_vectors = []

        for bs in probe_sizes:
            if bs > len(probe_images):
                break

            result = self._probe_batch_size(encoder, probe_images, bs)
            probes.append(result)

            if not result.success:
                # OOM → stop, previous batch size is the limit
                logger.info(f"[CALIBRATE] OOM at {bs}, limit is {optimal_bs}")
                break

            logger.info(
                f"[CALIBRATE]   bs={bs:>3d}: {result.time_per_image:.3f}s/img "
                f"(total {result.total_time:.2f}s)"
            )

            # Check degradation vs best
            if result.time_per_image < best_tpi:
                best_tpi = result.time_per_image
                optimal_bs = bs
            elif result.time_per_image > best_tpi * self.DEGRADATION_THRESHOLD:
                # Performance degraded > 15%
                logger.info(
                    f"[CALIBRATE] Degradation at bs={bs} "
                    f"({result.time_per_image:.3f} vs best {best_tpi:.3f}), "
                    f"optimal={optimal_bs}"
                )
                break

        # Apply safety margin
        recommended = max(1, int(optimal_bs * self.SAFETY_MARGIN))
        logger.info(
            f"[CALIBRATE] VV optimal={optimal_bs}, "
            f"recommended={recommended} (safety={self.SAFETY_MARGIN})"
        )

        # Encode probe images at the recommended batch size for reuse
        try:
            probe_count = min(len(probe_images), recommended)
            last_successful_vectors = encoder.encode_image_batch(
                probe_images[:probe_count], batch_size=recommended
            )
        except Exception:
            last_successful_vectors = []

        self._save_cache(recommended, optimal_bs, probes)

        return recommended, last_successful_vectors
