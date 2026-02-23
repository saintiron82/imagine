"""
Ollama Vision Adapter - Memory-efficient AI vision analysis.

This adapter uses Ollama for vision analysis with automatic memory management.
Models are loaded only when needed and unloaded immediately after use.

v3 P0: 2-Stage Pipeline support (classify → analyze_structured).
"""

import logging
import json
import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import requests

from .prompts import STAGE1_PROMPT, get_stage2_prompt
from .schemas import STAGE1_SCHEMA, get_schema
from .repair import parse_structured_output

logger = logging.getLogger(__name__)


class OllamaVisionAdapter:
    """
    Vision analyzer using Ollama with automatic memory management.

    Key features:
    - Automatic model loading/unloading (keep_alive=0)
    - Memory efficient (only uses RAM during analysis)
    - Works with local Ollama server
    """

    def __init__(
        self,
        model: str = None,
        host: str = None
    ):
        """
        Initialize Ollama vision adapter.

        Args:
            model: Ollama model name (reads from config.yaml if None)
            host: Ollama server URL (reads from config.yaml if None)
        """
        from ..utils.config import get_config
        cfg = get_config()

        self.model = model or cfg.get("vision.model", "qwen3-vl:8b")
        self.host = host or cfg.get("vision.ollama_host", "http://localhost:11434")
        self.api_url = f"{self.host}/api/generate"
        self._temperature = cfg.get("vision.temperature", 0.1)
        self._max_retries = cfg.get("vision.max_retries", 2)
        self._keep_alive = cfg.get("vision.keep_alive", "5m")

        logger.info(f"OllamaVisionAdapter initialized (model: {self.model}, host: {self.host})")

    def check_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using Ollama vision model.

        Args:
            image: PIL Image to analyze
            context: Optional context information

        Returns:
            Dictionary with analysis results:
            {
                "caption": str,
                "tags": list[str],
                "ocr": str,
                "color": str,
                "style": str
            }
        """
        if not self.check_ollama_running():
            logger.error("Ollama server is not running!")
            return self._empty_result()

        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Prepare prompt
            prompt = self._build_prompt(context)

            # Call Ollama API with keep_alive=0 for immediate memory release
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "keep_alive": 0  # 🔥 Key: Unload model immediately after use
            }

            logger.info(f"Analyzing image with {self.model}...")
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._empty_result()

            result = response.json()
            response_text = result.get("response", "")

            # Parse response
            parsed = self._parse_response(response_text)

            logger.info(f"Analysis complete: {len(parsed['tags'])} tags extracted")
            return parsed

        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._empty_result()

    def analyze_file(
        self,
        image_path: Path,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image file.

        Args:
            image_path: Path to image file
            context: Optional context information

        Returns:
            Analysis results dictionary
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.analyze(image, context)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return self._empty_result()

    def _build_prompt(self, context: Optional[Dict] = None) -> str:
        """Build analysis prompt."""
        prompt = """Analyze this image and provide:

1. A detailed caption (1-2 sentences describing what you see)
2. Extract 5-10 relevant tags/keywords
3. Any visible text (OCR)
4. Dominant color
5. Art style (if applicable)

Format your response as JSON:
{
    "caption": "description here",
    "tags": ["tag1", "tag2", ...],
    "ocr": "any visible text",
    "color": "dominant color name",
    "style": "art style description"
}"""

        if context and context.get("layer_name"):
            prompt += f"\n\nContext: This is a layer named '{context['layer_name']}'"

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Ollama response text.

        Tries to extract JSON, falls back to text parsing.
        """
        # Try JSON parsing first
        try:
            # Find JSON in response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)

                return {
                    "caption": data.get("caption", ""),
                    "tags": data.get("tags", []),
                    "ocr": data.get("ocr", ""),
                    "color": data.get("color", ""),
                    "style": data.get("style", "")
                }
        except json.JSONDecodeError:
            pass

        # Fallback: Extract from text
        lines = response_text.split('\n')
        caption = ""
        tags = []

        for line in lines:
            line = line.strip()
            if line and not caption:
                caption = line
            elif ',' in line:
                # Likely tags
                tags.extend([t.strip() for t in line.split(',') if t.strip()])

        return {
            "caption": caption or response_text[:200],
            "tags": tags[:10] if tags else self._extract_tags_simple(response_text),
            "ocr": "",
            "color": "",
            "style": ""
        }

    def _extract_tags_simple(self, text: str) -> list[str]:
        """Simple tag extraction from text."""
        words = text.lower().split()
        # Filter common words, keep meaningful nouns/adjectives
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        tags = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(tags))[:10]

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            "caption": "",
            "tags": [],
            "ocr": "",
            "color": "",
            "style": ""
        }

    # ── v3 P0: 2-Stage Pipeline ────────────────────────────

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 JPEG string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _call_ollama(self, prompt: str, img_base64: str, keep_alive: str = None) -> str:
        """Low-level Ollama API call. Returns raw response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "options": {"temperature": self._temperature},
            "keep_alive": keep_alive or self._keep_alive,
        }
        response = requests.post(self.api_url, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")
        return response.json().get("response", "")

    def classify(self, image: Image.Image, keep_alive: str = None, domain=None) -> Dict[str, Any]:
        """
        Stage 1: Classify image type.

        Args:
            image: PIL Image
            keep_alive: Model keep-alive duration
            domain: Optional DomainProfile for domain-scoped classification

        Returns:
            {"image_type": "character"|"background"|..., "confidence": "high"|...}
        """
        from .prompts import build_stage1_prompt

        if not self.check_ollama_running():
            logger.error("Ollama server is not running!")
            return {"image_type": "other", "confidence": "low"}

        prompt = build_stage1_prompt(domain) if domain else STAGE1_PROMPT
        img_b64 = self._encode_image(image)

        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                raw = self._call_ollama(prompt, img_b64, keep_alive)
                elapsed = time.perf_counter() - t0
                result = parse_structured_output(raw, STAGE1_SCHEMA, image_type="other")
                if result.get("image_type"):
                    logger.info(f"Stage 1 completed in {elapsed:.1f}s → {result.get('image_type')}")
                    return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning(f"Stage 1 attempt {attempt + 1} failed after {elapsed:.1f}s: {e}")

        return {"image_type": "other", "confidence": "low"}

    def analyze_structured(
        self, image: Image.Image, image_type: str, keep_alive: str = None,
        context: dict = None, domain=None
    ) -> Dict[str, Any]:
        """
        Stage 2: Type-specific structured analysis.

        Args:
            image: PIL Image
            image_type: Result from Stage 1 classify()
            keep_alive: Override keep_alive for batch processing
            context: Optional file metadata context
            domain: Optional DomainProfile for domain-specific hints

        Returns:
            Structured dict with type-specific fields + caption + tags
        """
        if not self.check_ollama_running():
            logger.error("Ollama server is not running!")
            return {"caption": "", "tags": [], "image_type": image_type}

        # v3.1: Inject context into Stage 2 prompt
        prompt = get_stage2_prompt(image_type, context=context, domain=domain)
        schema = get_schema(image_type)
        img_b64 = self._encode_image(image)

        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                raw = self._call_ollama(prompt, img_b64, keep_alive)
                elapsed = time.perf_counter() - t0
                result = parse_structured_output(raw, schema, image_type=image_type)
                result["image_type"] = image_type
                logger.info(f"Stage 2 completed in {elapsed:.1f}s ({image_type})")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning(f"Stage 2 attempt {attempt + 1} failed after {elapsed:.1f}s: {e}")

        return {"caption": "", "tags": [], "image_type": image_type}

    def classify_and_analyze(
        self, image: Image.Image, keep_alive: str = None, context: dict = None,
        domain=None
    ) -> Dict[str, Any]:
        """
        Full 2-Stage pipeline: classify → analyze_structured.

        Args:
            image: PIL Image to analyze
            keep_alive: Model keep-alive duration (default from config)
            context: Optional file metadata context (v3.1: MC.raw)
            domain: Optional DomainProfile for domain-aware classification

        Returns:
            Merged dict with image_type + all structured fields
        """
        t_total = time.perf_counter()

        classification = self.classify(image, keep_alive, domain=domain)
        image_type = classification.get("image_type", "other")
        logger.info(f"Stage 1 → {image_type} (confidence: {classification.get('confidence', '?')})")

        analysis = self.analyze_structured(image, image_type, keep_alive, context=context, domain=domain)

        total_elapsed = time.perf_counter() - t_total
        logger.info(f"2-Stage total: {total_elapsed:.1f}s ({image_type})")

        return analysis

    def classify_and_analyze_sequence(
        self, items: list, progress_callback=None, domain=None
    ) -> list:
        """Process multiple images sequentially via Ollama API."""
        results = []
        for idx, (image, context) in enumerate(items):
            try:
                result = self.classify_and_analyze(image, context=context, domain=domain)
            except Exception as e:
                logger.warning(f"Vision failed for item {idx}: {e}")
                result = {"caption": "", "tags": [], "image_type": "other"}
            results.append(result)
            if progress_callback:
                progress_callback(idx, len(items), result)
        return results

    def unload_model(self):
        """Explicitly unload model from VRAM after batch processing."""
        try:
            payload = {
                "model": self.model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            }
            requests.post(self.api_url, json=payload, timeout=10)
            logger.info(f"Model {self.model} unloaded")
        except Exception:
            pass


if __name__ == "__main__":
    # Test
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python ollama_adapter.py <image_path>")
        sys.exit(1)

    adapter = OllamaVisionAdapter()
    result = adapter.analyze_file(Path(sys.argv[1]))

    print("\n" + "="*80)
    print("Analysis Result:")
    print("="*80)
    print(f"Caption: {result['caption']}")
    print(f"Tags: {', '.join(result['tags'])}")
    print(f"OCR: {result['ocr']}")
    print(f"Color: {result['color']}")
    print(f"Style: {result['style']}")
