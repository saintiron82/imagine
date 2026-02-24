"""
MLX Vision Adapter - Native Apple Silicon VLM backend.

Uses mlx-vlm to run Qwen3-VL on Apple Silicon with native Metal acceleration.
Provides 3-4x TTFT improvement and ~30x faster token generation vs PyTorch MPS.

v6.4: Initial MLX backend integration via Factory Pattern.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from PIL import Image

logger = logging.getLogger(__name__)


class MLXVisionAdapter:
    """
    VLM adapter using mlx-vlm for native Apple Silicon inference.

    Implements the same interface as VisionAnalyzer (Transformers) and
    OllamaVisionAdapter so it can be used interchangeably via VisionAnalyzerFactory.
    """

    def __init__(
        self,
        model: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        tier_name: str = "pro",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """
        Initialize MLX Vision Adapter.

        Args:
            model: MLX model ID (HuggingFace mlx-community path)
            tier_name: AI tier for metadata tracking
            max_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.model_id = model
        self.tier_name = tier_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._model = None
        self._processor = None
        self._config = None

        logger.info(
            f"MLXVisionAdapter initialized (tier: {tier_name}, model: {model}, "
            f"max_tokens: {max_tokens}, temperature: {temperature})"
        )

    def _load_model(self):
        """Lazy load the MLX VLM model."""
        if self._model is not None:
            return

        logger.info(f"Loading MLX VLM model: {self.model_id}")
        t0 = time.perf_counter()

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            self._model, self._processor = load(self.model_id)
            self._config = load_config(self.model_id)

            elapsed = time.perf_counter() - t0
            logger.info(f"MLX VLM model loaded in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load MLX VLM model {self.model_id}: {e}")
            raise

    def _generate_response(
        self, image: Image.Image, user_prompt: str, system_prompt: str = None
    ) -> str:
        """
        Generate text response from VLM using mlx-vlm.

        Args:
            image: PIL Image
            user_prompt: Text prompt for the model
            system_prompt: Optional system role message

        Returns:
            Model-generated text
        """
        self._load_model()

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Build prompt with chat template
        prompt = user_prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_prompt}"

        formatted_prompt = apply_chat_template(
            self._processor, self._config, prompt, num_images=1
        )

        # Generate response (returns GenerationResult with .text attribute)
        result = generate(
            self._model,
            self._processor,
            formatted_prompt,
            [image],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        )

        # Extract text from GenerationResult object
        if hasattr(result, 'text'):
            return result.text
        return str(result)

    # ── Legacy single-pass interface ──────────────────────────────

    def analyze(
        self, image: Image.Image, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Legacy single-pass analysis. Delegates to 2-Stage pipeline."""
        return self.classify_and_analyze(image, context=context)

    def analyze_file(
        self, image_path, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze an image file."""
        try:
            image = Image.open(image_path).convert("RGB")
            return self.analyze(image, context)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return {"caption": "", "tags": [], "image_type": "other"}

    # ── 2-Stage Pipeline ──────────────────────────────────────────

    def classify(
        self, image: Image.Image, keep_alive: str = None, domain=None
    ) -> Dict[str, Any]:
        """
        Stage 1: Classify image type.

        Args:
            image: PIL Image
            keep_alive: Ignored (interface compatibility)
            domain: Optional DomainProfile for domain-scoped classification

        Returns:
            {"image_type": str, "confidence": str}
        """
        from .prompts import STAGE1_USER, STAGE1_SYSTEM, build_stage1_prompt
        from .schemas import STAGE1_SCHEMA
        from .repair import parse_structured_output

        prompt = build_stage1_prompt(domain) if domain else STAGE1_USER
        try:
            t0 = time.perf_counter()
            raw = self._generate_response(image, prompt, STAGE1_SYSTEM)
            elapsed = time.perf_counter() - t0
            result = parse_structured_output(raw, STAGE1_SCHEMA, image_type="other")
            logger.info(
                f"[MLX] Stage 1 completed in {elapsed:.1f}s → {result.get('image_type')}"
            )
            return result
        except Exception as e:
            logger.warning(f"[MLX] Stage 1 classification failed: {e}")
            return {"image_type": "other", "confidence": "low"}

    def analyze_structured(
        self,
        image: Image.Image,
        image_type: str,
        keep_alive: str = None,
        context: dict = None,
        domain=None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Type-specific structured analysis.

        Args:
            image: PIL Image
            image_type: Result from Stage 1
            keep_alive: Ignored (interface compatibility)
            context: Optional file metadata context
            domain: Optional DomainProfile for domain-specific hints

        Returns:
            Structured dict with type-specific fields + caption + tags
        """
        from .prompts import get_stage2_prompt, STAGE2_SYSTEM
        from .schemas import get_schema
        from .repair import parse_structured_output

        try:
            t0 = time.perf_counter()
            prompt = get_stage2_prompt(image_type, context=context, domain=domain)
            raw = self._generate_response(image, prompt, STAGE2_SYSTEM)
            elapsed = time.perf_counter() - t0
            result = parse_structured_output(
                raw, get_schema(image_type), image_type=image_type
            )
            result["image_type"] = image_type
            logger.info(f"[MLX] Stage 2 completed in {elapsed:.1f}s ({image_type})")
            return result
        except Exception as e:
            logger.warning(f"[MLX] Stage 2 analysis failed: {e}")
            return {"caption": "", "tags": [], "image_type": image_type}

    def classify_and_analyze(
        self, image: Image.Image, keep_alive: str = None, context: dict = None,
        domain=None
    ) -> Dict[str, Any]:
        """
        Full 2-Stage pipeline: classify -> analyze_structured.

        Args:
            image: PIL Image to analyze
            keep_alive: Ignored (interface compatibility)
            context: Optional file metadata context
            domain: Optional DomainProfile for domain-aware classification

        Returns:
            Merged dict with image_type + all structured fields
        """
        t_total = time.perf_counter()

        classification = self.classify(image, keep_alive, domain=domain)
        image_type = classification.get("image_type", "other")
        logger.info(
            f"[MLX] Stage 1 → {image_type} "
            f"(confidence: {classification.get('confidence', '?')})"
        )

        analysis = self.analyze_structured(
            image, image_type, keep_alive, context=context, domain=domain
        )

        total_elapsed = time.perf_counter() - t_total
        logger.info(f"[MLX] 2-Stage total: {total_elapsed:.1f}s ({image_type})")

        return analysis

    def classify_and_analyze_sequence(
        self,
        items: list,
        progress_callback=None,
        domain=None,
    ) -> list:
        """
        Process multiple images sequentially through 2-stage pipeline.

        MLX processes images sequentially (no batch API), but individual
        image inference is fast enough due to native Apple Silicon acceleration.

        Args:
            items: List of (PIL Image, context dict) tuples
            progress_callback: Optional fn(index, total, result)

        Returns:
            List of vision result dicts (same order as input)
        """
        self._load_model()

        results = []
        for idx, (image, context) in enumerate(items):
            try:
                result = self.classify_and_analyze(image, context=context, domain=domain)
            except Exception as e:
                logger.warning(f"[MLX] Vision failed for item {idx}: {e}")
                result = {"caption": "", "tags": [], "image_type": "other"}
            results.append(result)
            if progress_callback:
                progress_callback(idx, len(items), result)
        return results

    def unload_model(self):
        """Unload MLX model to free memory.

        MLX uses its own Metal buffer allocator (not torch.mps),
        so we must call mx.clear_cache() to release GPU memory.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._config is not None:
            del self._config
            self._config = None
        gc.collect()
        # MLX metal cache must be cleared explicitly — torch.mps.empty_cache()
        # has no effect on MLX allocations.
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        logger.info(f"[MLX] Model unloaded ({self.model_id})")
