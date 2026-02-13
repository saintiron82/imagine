"""
SigLIP 2 NaFlex VV (Visual Vector) Encoder.

Uses SigLIP 2 So400m NaFlex (1152-dim) with dynamic resolution.
NaFlex preserves native aspect ratio for better embedding quality.

v3.3: Adaptive batch sizing with runtime discovery and fine-tuning.
      Replaces fixed calibration with dynamic adaptation inside
      encode_image_batch itself.

API: get_image_features / get_text_features → pooler_output (768/1152/1664,)
"""

import base64
import gc
import json
import logging
import platform
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# OOM error patterns (CUDA + MPS + generic)
OOM_PATTERNS = [
    "out of memory",
    "cuda out of memory",
    "mps backend",
    "mps out of memory",
    "allocation",
    "allocate",
]

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration_results"


class SigLIP2Encoder:
    """
    Lazy-loaded SigLIP 2 NaFlex encoder for VV generation.

    v3.3: Adaptive batch sizing — discovers optimal batch size at runtime
    using probe → scale-up → fine-tune, with OOM recovery and cache hints.

    Usage:
        encoder = SigLIP2Encoder()                        # config-driven
        img_vec = encoder.encode_image(pil_image)          # np.ndarray (1152,)
        vectors = encoder.encode_image_batch(images)       # adaptive batch
        txt_vec = encoder.encode_text("search query")     # np.ndarray (1152,)
    """

    # Adaptive batch constants
    INITIAL_BATCH_SIZE = 64         # Starting probe size (no cache)
    FINE_TUNE_GAP = 3               # Min gap to trigger fine-tune
    DEGRADATION_THRESHOLD = 1.15    # 15% slower = degradation
    MIN_DISCOVERY_IMAGES = 4        # Need at least 4 images for discovery

    def __init__(self, model_name: Optional[str] = None):
        from ..utils.config import get_config
        from ..utils.tier_config import get_active_tier

        cfg = get_config()

        # v3.1: Tier-aware model selection
        tier_name, tier_config = get_active_tier()
        visual_config = tier_config.get("visual", {})

        # Priority: explicit param > tier config > legacy config
        self.model_name = model_name or visual_config.get("model") or cfg.get(
            "embedding.visual.model", "google/siglip2-so400m-patch16-naflex"
        )
        self._dimensions = visual_config.get("dimensions") or cfg.get(
            "embedding.visual.dimensions", 1152
        )
        self.tier_name = tier_name
        self._model = None
        self._processor = None   # Image processor
        self._tokenizer = None   # Text tokenizer (for encode_text)
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        # Adaptive batch state (persists across encode_image_batch calls)
        self._adaptive_bs: Optional[int] = None
        self._oom_boundary: Optional[int] = None

        logger.info(
            f"SigLIP2Encoder initialized (tier: {tier_name}, model: {self.model_name}, "
            f"dimensions: {self._dimensions}, device: {self._device})"
        )

    def _load(self):
        """Lazy-load model, image processor, and tokenizer on first use."""
        if self._model is not None and self._processor is not None:
            return

        from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

        logger.info(f"Loading SigLIP 2 model: {self.model_name}...")
        try:
            self._model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.float16, local_files_only=True
            ).to(self._device).eval()
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            logger.info(f"SigLIP 2 model loaded ({self._device}, fp16)")
        except OSError:
            logger.info("Local cache not found, downloading from HuggingFace...")
            self._model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self._device).eval()
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"SigLIP 2 model downloaded and loaded ({self._device}, fp16)")
        except Exception as e:
            logger.error(f"Failed to load SigLIP 2 model: {e}")
            self._model = None
            self._processor = None
            self._tokenizer = None
            raise

    def unload(self):
        """Unload SigLIP2 model to free GPU memory."""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._adaptive_bs = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info(f"SigLIP2 model unloaded ({self.model_name})")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # ── OOM Utilities ──────────────────────────────────────────────────

    @staticmethod
    def _is_oom(error: Exception) -> bool:
        msg = str(error).lower()
        return any(p in msg for p in OOM_PATTERNS)

    def _clear_gpu_cache(self):
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Cache (hint-based, not authoritative) ──────────────────────────

    def _cache_path(self) -> Path:
        os_name = platform.system().lower()
        model_short = self.model_name.split("/")[-1][:30].replace(" ", "-").lower()
        return CALIBRATION_DIR / (
            f"adaptive_vv_{os_name}_{self.tier_name}_"
            f"{model_short}_{self._device}.json"
        )

    def _load_cache_hint(self) -> Optional[int]:
        """Load cached optimal as starting hint (50% of cached value)."""
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if (
                data.get("model") == self.model_name
                and data.get("device") == self._device
                and data.get("tier") == self.tier_name
            ):
                cached = data.get("optimal_batch_size")
                if cached and cached > 0:
                    hint = max(1, cached // 2)
                    logger.info(
                        f"[ADAPTIVE] Cache hint: optimal was {cached}, "
                        f"starting from {hint}"
                    )
                    return hint
        except Exception:
            pass
        return None

    def _save_cache(self, optimal: int):
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "optimal_batch_size": optimal,
            "model": self.model_name,
            "device": self._device,
            "tier": self.tier_name,
            "platform": platform.system().lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self._cache_path()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"[ADAPTIVE] Saved cache: optimal={optimal}")

    # ── Forward Pass ───────────────────────────────────────────────────

    def _encode_forward(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Pure batch forward pass — no error handling, raises on OOM.
        Returns L2-normalized vectors.
        """
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        if hasattr(image_features, "pooler_output"):
            batch_vecs = image_features.pooler_output
        elif image_features.dim() == 2:
            batch_vecs = image_features
        else:
            batch_vecs = image_features[1]

        vectors = []
        for j in range(batch_vecs.shape[0]):
            vec = batch_vecs[j].float().cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.astype(np.float32))

        # Free GPU tensors immediately (MPS doesn't auto-reclaim)
        del inputs, image_features, batch_vecs
        if self._device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return vectors

    # ── Adaptive Discovery ─────────────────────────────────────────────

    def _discover_optimal_batch(
        self, images: List[Image.Image]
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Discover optimal batch size: probe → scale-up → fine-tune.

        Uses actual images for probing so discovery vectors can be reused.

        Returns:
            (optimal_batch_size, vectors_for_first_N_images)
        """
        n = len(images)

        if n < self.MIN_DISCOVERY_IMAGES:
            bs = min(4, n)
            self._adaptive_bs = bs
            logger.info(
                f"[ADAPTIVE] Too few images ({n}) for discovery, using bs={bs}"
            )
            return bs, []

        # Starting point: cache hint or INITIAL_BATCH_SIZE
        hint = self._load_cache_hint()
        start = min(hint or self.INITIAL_BATCH_SIZE, n)

        logger.info(
            f"[ADAPTIVE] Discovering optimal VV batch "
            f"(start={start}, images={n})..."
        )

        # ── Phase 1: Find a working batch size (halve on OOM) ──
        working_bs = None
        oom_bs = None
        working_vecs: List[np.ndarray] = []
        working_tpi = float("inf")

        current = start
        while current >= 1:
            probe = images[:current]
            t0 = time.perf_counter()
            try:
                vecs = self._encode_forward(probe)
                elapsed = time.perf_counter() - t0
                tpi = elapsed / len(probe)
                working_bs = current
                working_vecs = vecs
                working_tpi = tpi
                logger.info(
                    f"[ADAPTIVE]   bs={current:>3d}: OK "
                    f"({tpi:.3f}s/img, total {elapsed:.2f}s)"
                )
                break
            except RuntimeError as e:
                if self._is_oom(e):
                    oom_bs = current
                    self._clear_gpu_cache()
                    logger.info(f"[ADAPTIVE]   bs={current:>3d}: OOM")
                    current = max(1, current // 2)
                else:
                    raise

        if working_bs is None:
            logger.warning("[ADAPTIVE] All sizes failed, using bs=1")
            self._adaptive_bs = 1
            return 1, []

        # ── Phase 2: Scale up (if no OOM boundary yet) ──
        if oom_bs is None and working_bs < n:
            scale_sizes = []
            s = working_bs * 2
            while s <= min(n, 256):
                scale_sizes.append(s)
                s *= 2

            for bs in scale_sizes:
                probe = images[:min(bs, n)]
                t0 = time.perf_counter()
                try:
                    vecs = self._encode_forward(probe)
                    elapsed = time.perf_counter() - t0
                    tpi = elapsed / len(probe)
                    logger.info(
                        f"[ADAPTIVE]   bs={bs:>3d}: OK ({tpi:.3f}s/img)"
                    )

                    if tpi > working_tpi * self.DEGRADATION_THRESHOLD:
                        logger.info(
                            f"[ADAPTIVE]   Degraded at bs={bs} "
                            f"({tpi:.3f} vs {working_tpi:.3f}), stopping"
                        )
                        oom_bs = bs  # Soft boundary
                        break

                    working_bs = bs
                    working_vecs = vecs
                    working_tpi = tpi

                except RuntimeError as e:
                    if self._is_oom(e):
                        oom_bs = bs
                        self._clear_gpu_cache()
                        logger.info(f"[ADAPTIVE]   bs={bs:>3d}: OOM")
                        break
                    raise

        # ── Phase 3: Fine-tune between working and boundary ──
        if oom_bs is not None and oom_bs - working_bs > self.FINE_TUNE_GAP:
            fine_bs, fine_vecs = self._fine_tune(
                images, working_bs, oom_bs, working_tpi
            )
            if fine_bs > working_bs:
                working_bs = fine_bs
                if fine_vecs:
                    working_vecs = fine_vecs

        self._adaptive_bs = working_bs
        self._oom_boundary = oom_bs
        self._save_cache(working_bs)

        logger.info(f"[ADAPTIVE] Optimal batch size: {working_bs}")
        return working_bs, working_vecs

    def _fine_tune(
        self,
        images: List[Image.Image],
        working: int,
        oom: int,
        working_tpi: float,
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Binary search between working and OOM boundary.

        Tests intermediate batch sizes to find the exact optimal.
        Returns (best_batch_size, vectors_from_best_probe).
        """
        logger.info(
            f"[ADAPTIVE] Fine-tuning between {working} and {oom}..."
        )

        best = working
        best_tpi = working_tpi
        best_vecs: List[np.ndarray] = []

        lo, hi = working + 1, oom - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            n_probe = min(mid, len(images))
            if n_probe < mid:
                hi = mid - 1
                continue

            probe = images[:n_probe]
            t0 = time.perf_counter()
            try:
                vecs = self._encode_forward(probe)
                elapsed = time.perf_counter() - t0
                tpi = elapsed / n_probe
                logger.info(
                    f"[ADAPTIVE] [FINE-TUNE] bs={mid}: OK ({tpi:.3f}s/img)"
                )

                if tpi <= best_tpi * self.DEGRADATION_THRESHOLD:
                    best = mid
                    best_vecs = vecs
                    if tpi < best_tpi:
                        best_tpi = tpi
                    lo = mid + 1
                else:
                    logger.info(
                        f"[ADAPTIVE] [FINE-TUNE] bs={mid}: degraded"
                    )
                    hi = mid - 1

            except RuntimeError as e:
                if self._is_oom(e):
                    logger.info(
                        f"[ADAPTIVE] [FINE-TUNE] bs={mid}: OOM"
                    )
                    self._clear_gpu_cache()
                    hi = mid - 1
                else:
                    raise

        logger.info(f"[ADAPTIVE] Fine-tune result: {working} → {best}")
        return best, best_vecs

    # ── Public API ─────────────────────────────────────────────────────

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image to a normalized embedding vector.

        NaFlex preserves native aspect ratio with dynamic patch count.

        Args:
            image: PIL Image (RGB)

        Returns:
            L2-normalized numpy array of shape (1152,)
        """
        self._load()
        inputs = self._processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        if hasattr(image_features, 'pooler_output'):
            vec = image_features.pooler_output[0]
        elif image_features.dim() == 2:
            vec = image_features[0]
        else:
            vec = image_features[1][0]

        vec = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_image_batch(
        self, images: List[Image.Image], batch_size: int = None
    ) -> List[np.ndarray]:
        """
        Encode multiple PIL Images with adaptive batch sizing.

        v3.3: Automatically discovers optimal batch size on first call
        via probe → scale-up → fine-tune. Uses cache hints for fast
        subsequent sessions. Recovers from OOM by halving.

        Args:
            images: List of PIL Images (RGB)
            batch_size: Force a specific batch size (None=adaptive)

        Returns:
            List of L2-normalized numpy arrays of shape (dimensions,)
        """
        if not images:
            return []

        self._load()
        all_vectors: List[np.ndarray] = []
        start_idx = 0

        # Determine effective batch size
        if batch_size is not None:
            # Explicit: use it, still with OOM recovery
            effective_bs = batch_size
        elif self._device == "cpu":
            effective_bs = 4
            self._adaptive_bs = 4
        elif self._adaptive_bs is not None:
            # Reuse previously discovered optimal
            effective_bs = self._adaptive_bs
        else:
            # First call: discover optimal (vectors reused from probes)
            effective_bs, discovery_vecs = self._discover_optimal_batch(images)
            if discovery_vecs:
                all_vectors.extend(discovery_vecs)
                start_idx = len(discovery_vecs)
                if start_idx >= len(images):
                    logger.info(
                        f"VV batch encoded {len(all_vectors)} images "
                        f"(adaptive_bs={self._adaptive_bs})"
                    )
                    return all_vectors

        # Process remaining images in sub-batches
        i = start_idx
        while i < len(images):
            sub = images[i: i + effective_bs]

            try:
                vecs = self._encode_forward(sub)
                all_vectors.extend(vecs)
                i += len(sub)

            except RuntimeError as e:
                if self._is_oom(e) and len(sub) > 1:
                    # Runtime OOM: halve and re-adapt
                    effective_bs = max(1, len(sub) // 2)
                    self._adaptive_bs = effective_bs
                    self._clear_gpu_cache()
                    logger.warning(
                        f"[ADAPTIVE] Runtime OOM, reducing to bs={effective_bs}"
                    )
                    continue  # Retry with smaller batch

                elif self._is_oom(e):
                    # Single image OOM: zero vector fallback
                    logger.warning(
                        "[ADAPTIVE] Single image OOM, using zero vector"
                    )
                    all_vectors.append(
                        np.zeros(self._dimensions, dtype=np.float32)
                    )
                    i += 1
                    self._clear_gpu_cache()
                else:
                    raise

            except Exception as e:
                # Non-RuntimeError: single-image fallback for this sub-batch
                logger.warning(
                    f"VV sub-batch error ({len(sub)} images): {e}, "
                    f"falling back to single-image"
                )
                for img in sub:
                    try:
                        all_vectors.append(self.encode_image(img))
                    except Exception:
                        all_vectors.append(
                            np.zeros(self._dimensions, dtype=np.float32)
                        )
                i += len(sub)

        logger.info(
            f"VV batch encoded {len(all_vectors)} images "
            f"(adaptive_bs={self._adaptive_bs})"
        )
        return all_vectors

    def encode_image_from_base64(self, base64_data: str) -> np.ndarray:
        """
        Encode a base64-encoded image to a normalized embedding vector.

        Accepts raw base64 or data URL (data:image/...;base64,...).

        Args:
            base64_data: Base64-encoded image string

        Returns:
            L2-normalized numpy array of shape (dimensions,)
        """
        if "," in base64_data and base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]

        raw = base64.b64decode(base64_data)
        image = Image.open(BytesIO(raw)).convert("RGB")
        return self.encode_image(image)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query to a normalized embedding vector.

        Processor handles lowercasing + padding automatically.

        Args:
            text: Search query string

        Returns:
            L2-normalized numpy array of shape (1152,)
        """
        self._load()
        inputs = self._tokenizer(
            [text], return_tensors="pt",
            padding="max_length", max_length=64, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

        if hasattr(text_features, 'pooler_output'):
            vec = text_features.pooler_output[0]
        elif text_features.dim() == 2:
            vec = text_features[0]
        else:
            vec = text_features[1][0]

        vec = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
