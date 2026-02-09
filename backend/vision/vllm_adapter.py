"""
vLLM Vision Adapter - High-performance batch processing.

This adapter uses vLLM for vision analysis with optimized batch processing.
vLLM provides 3-5x faster inference compared to Ollama through:
- PagedAttention (efficient memory management)
- Continuous Batching (dynamic batch processing)
- Optimized CUDA kernels

**Platform Support**: Mac and Linux only (Unix-like systems)
**Windows**: Not supported (use Ollama or Transformers instead)

v3.1.1: Cross-platform optimization for ImageParser
"""

import logging
import platform
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

from .prompts import STAGE1_PROMPT, get_stage2_prompt
from .schemas import STAGE1_SCHEMA, get_schema
from .repair import parse_structured_output

logger = logging.getLogger(__name__)


class VLLMAdapter:
    """
    Vision analyzer using vLLM with optimized batch processing.

    Key features:
    - 3-5x faster than Ollama
    - Excellent batch processing (16+ images simultaneously)
    - PagedAttention for memory efficiency
    - Continuous Batching for dynamic workloads

    Platform support:
    - ✓ Mac (Darwin)
    - ✓ Linux
    - ✗ Windows (not supported)
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        tier_name: str = "ultra",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None
    ):
        """
        Initialize vLLM vision adapter.

        Args:
            model: HuggingFace model ID (e.g., "Qwen/Qwen3-VL-8B-Instruct")
            tier_name: AI tier name for logging
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: GPU memory usage ratio (0.0-1.0)
            max_model_len: Maximum sequence length (None for model default)

        Raises:
            RuntimeError: If platform is Windows or vLLM is not installed
        """
        # Platform check
        if platform.system() == 'Windows':
            raise RuntimeError(
                "vLLM is not supported on Windows. "
                "Please use Ollama or Transformers backend instead."
            )

        # Check vLLM installation
        try:
            import vllm
            from vllm import LLM, SamplingParams
            logger.info(f"vLLM version: {vllm.__version__}")
        except ImportError:
            raise RuntimeError(
                "vLLM is not installed. "
                "Install it with: pip install vllm"
            )

        self.model_name = model
        self.tier_name = tier_name

        # Initialize vLLM engine
        logger.info(f"Initializing vLLM with {model} (tier: {tier_name})")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")

        try:
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True  # Required for Qwen models
            )

            # Default sampling parameters
            self.default_sampling = SamplingParams(
                temperature=0.1,
                max_tokens=1024,
                top_p=0.9
            )

            logger.info(f"✓ vLLM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _create_vision_prompt(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        """
        Create vLLM vision prompt format.

        vLLM expects prompts in a specific format for vision models.
        """
        return {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_base64
            }
        }

    def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Legacy single-pass analysis.

        Args:
            image: PIL Image to analyze
            context: Optional context information

        Returns:
            Dictionary with analysis results
        """
        logger.warning("analyze() is legacy method. Use classify_and_analyze() instead.")

        # For now, use 2-stage pipeline
        return self.classify_and_analyze(image, keep_alive=None)

    def analyze_file(self, image_path: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze an image file."""
        image = Image.open(image_path)
        return self.analyze(image, context)

    def classify(
        self,
        image: Image.Image,
        keep_alive: str = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Classify image type.

        Args:
            image: PIL Image
            keep_alive: Ignored (vLLM doesn't have keep_alive concept)

        Returns:
            {
                "image_type": str,
                "confidence": float,
                "reasoning": str
            }
        """
        try:
            # Prepare prompt
            img_base64 = self._image_to_base64(image)
            prompt_text = f"{STAGE1_PROMPT}\n\nOutput JSON only:"

            # vLLM inference
            from vllm import SamplingParams
            sampling = SamplingParams(
                temperature=0.1,
                max_tokens=512
            )

            # Single request
            outputs = self.llm.generate(
                prompts=[prompt_text],
                sampling_params=sampling,
                multi_modal_data={
                    "image": [img_base64]
                }
            )

            # Parse output
            raw_text = outputs[0].outputs[0].text.strip()
            logger.debug(f"Stage 1 raw output: {raw_text}")

            result = parse_structured_output(raw_text, STAGE1_SCHEMA)

            return {
                "image_type": result.get("image_type", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"Stage 1 classification failed: {e}")
            return {
                "image_type": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

    def analyze_structured(
        self,
        image: Image.Image,
        image_type: str,
        keep_alive: str = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Type-specific structured analysis.

        Args:
            image: PIL Image
            image_type: Image type from Stage 1
            keep_alive: Ignored

        Returns:
            Structured analysis result (schema depends on image_type)
        """
        try:
            # Get type-specific prompt and schema
            prompt_text = get_stage2_prompt(image_type)
            schema = get_schema(image_type)

            # Prepare image
            img_base64 = self._image_to_base64(image)

            # vLLM inference
            from vllm import SamplingParams
            sampling = SamplingParams(
                temperature=0.1,
                max_tokens=1024
            )

            outputs = self.llm.generate(
                prompts=[prompt_text],
                sampling_params=sampling,
                multi_modal_data={
                    "image": [img_base64]
                }
            )

            # Parse output
            raw_text = outputs[0].outputs[0].text.strip()
            logger.debug(f"Stage 2 raw output: {raw_text}")

            result = parse_structured_output(raw_text, schema)

            # Add metadata
            result["mc_tier"] = self.tier_name
            result["mc_backend"] = "vllm"
            result["mc_model"] = self.model_name

            return result

        except Exception as e:
            logger.error(f"Stage 2 analysis failed: {e}")
            return {
                "error": str(e),
                "mc_tier": self.tier_name,
                "mc_backend": "vllm"
            }

    def classify_and_analyze(
        self,
        image: Image.Image,
        keep_alive: str = None
    ) -> Dict[str, Any]:
        """
        Full 2-Stage pipeline: classify → analyze_structured.

        This is the recommended method for analysis.

        Args:
            image: PIL Image
            keep_alive: Ignored

        Returns:
            Combined result from both stages
        """
        # Stage 1: Classification
        stage1_result = self.classify(image, keep_alive)
        image_type = stage1_result.get("image_type", "unknown")

        logger.info(f"Stage 1: {image_type} (confidence: {stage1_result.get('confidence', 0.0):.2f})")

        # Stage 2: Structured analysis
        stage2_result = self.analyze_structured(image, image_type, keep_alive)

        # Merge results
        result = {**stage2_result}
        result["mc_classification"] = stage1_result

        return result

    def process_batch(
        self,
        images: List[Image.Image],
        image_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch processing (vLLM optimized).

        This method leverages vLLM's continuous batching for maximum performance.

        Args:
            images: List of PIL Images
            image_types: Optional pre-classified types (if None, will classify first)

        Returns:
            List of analysis results
        """
        if not images:
            return []

        logger.info(f"Batch processing {len(images)} images (vLLM optimized)")

        # Stage 1: Batch classification (if needed)
        if image_types is None:
            logger.info("Stage 1: Batch classification")
            image_types = []

            # Prepare batch prompts
            prompts = [f"{STAGE1_PROMPT}\n\nOutput JSON only:" for _ in images]
            img_base64_list = [self._image_to_base64(img) for img in images]

            # Batch inference
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=0.1, max_tokens=512)

            outputs = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling,
                multi_modal_data={
                    "image": img_base64_list
                }
            )

            # Parse classifications
            for output in outputs:
                raw_text = output.outputs[0].text.strip()
                result = parse_structured_output(raw_text, STAGE1_SCHEMA)
                image_types.append(result.get("image_type", "unknown"))

            logger.info(f"✓ Classified {len(image_types)} images")

        # Stage 2: Batch structured analysis
        logger.info("Stage 2: Batch structured analysis")

        # Group by image type for schema consistency
        type_groups = {}
        for idx, (img, img_type) in enumerate(zip(images, image_types)):
            if img_type not in type_groups:
                type_groups[img_type] = []
            type_groups[img_type].append((idx, img))

        # Process each type group
        results = [None] * len(images)

        for img_type, group in type_groups.items():
            indices, imgs = zip(*group)

            # Prepare batch
            prompt_text = get_stage2_prompt(img_type)
            prompts = [prompt_text for _ in imgs]
            img_base64_list = [self._image_to_base64(img) for img in imgs]

            # Batch inference
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=0.1, max_tokens=1024)

            outputs = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling,
                multi_modal_data={
                    "image": img_base64_list
                }
            )

            # Parse results
            schema = get_schema(img_type)
            for idx, output in zip(indices, outputs):
                raw_text = output.outputs[0].text.strip()
                result = parse_structured_output(raw_text, schema)
                result["mc_tier"] = self.tier_name
                result["mc_backend"] = "vllm"
                result["mc_model"] = self.model_name
                results[idx] = result

        logger.info(f"✓ Batch processing completed ({len(results)} results)")
        return results

    def unload_model(self):
        """
        Unload vLLM model from VRAM.

        Note: vLLM doesn't provide explicit unload.
        You need to delete the LLM instance and call gc.collect().
        """
        logger.info("vLLM doesn't support explicit model unloading. Use del llm; gc.collect()")
        pass
