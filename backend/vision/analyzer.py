"""
Vision Analyzer - Core AI analysis engine using Florence-2.

This module analyzes images to extract:
- Detailed captions
- Tags and keywords
- OCR text
- Objects and their locations
- Style and color information
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """
    AI-powered vision analyzer using Florence-2 model.

    Florence-2 is a lightweight vision language model by Microsoft that
    can perform multiple vision tasks including captioning, OCR, and
    object detection.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_id: Optional[str] = None,
        dtype: str = "float16",
        tier_name: str = "pro"
    ):
        """
        Initialize the vision analyzer.

        Args:
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            model_id: Model to use (default: Qwen2-VL-2B-Instruct)
            dtype: Model precision ('float16', 'bfloat16', 'float32')
            tier_name: AI tier for metadata tracking ('standard', 'pro', 'ultra')
        """
        if device and device != 'auto':
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        # Default model: BLIP for stability, Qwen2-VL for quality
        # Options: "Salesforce/blip-image-captioning-large", "Qwen/Qwen2-VL-2B-Instruct"
        self.model_id = model_id if model_id else "Qwen/Qwen2-VL-2B-Instruct"
        self.dtype = dtype
        self.tier_name = tier_name
        self.model = None
        self.processor = None

        logger.info(
            f"VisionAnalyzer initialized (tier: {tier_name}, device: {self.device}, "
            f"model: {self.model_id}, dtype: {dtype})"
        )

    def _load_model(self):
        """Lazy load the vision model."""
        if self.model is not None:
            return

        logger.info(f"Loading {self.model_id} model...")

        try:
            if "blip2" in self.model_id.lower():
                # BLIP-2 model
                from transformers import Blip2Processor, Blip2ForConditionalGeneration

                self.processor = Blip2Processor.from_pretrained(self.model_id)

                # Use float16 on CUDA for speed, float32 on CPU
                dtype = torch.float16 if self.device == "cuda" else torch.float32

                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype
                ).to(self.device)

            elif "blip" in self.model_id.lower():
                # BLIP (original) model (default, simple and reliable)
                from transformers import BlipProcessor, BlipForConditionalGeneration

                self.processor = BlipProcessor.from_pretrained(self.model_id)

                # Use float16 on CUDA for speed, float32 on CPU
                dtype = torch.float16 if self.device == "cuda" else torch.float32

                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype
                ).to(self.device)

            elif "Qwen2-VL" in self.model_id:
                # Qwen2-VL model
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )

            elif "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id:
                # Qwen3-VL model (Standard/Pro/Ultra tiers)
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
                from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

                self.processor = Qwen3VLProcessor.from_pretrained(self.model_id)

                # MPS does not support device_map="auto"
                if self.device == "mps":
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                    ).to(self.device)
                else:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto"
                    )

            else:
                # Florence-2 or other models
                from transformers import AutoProcessor, AutoModelForCausalLM

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)

            logger.info(f"{self.model_id} loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load {self.model_id}: {e}")
            raise

    def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image and extract comprehensive information.

        Args:
            image: PIL Image object to analyze
            context: Optional context information (layer name, type, etc.)

        Returns:
            Dictionary containing:
                - caption: Detailed description
                - tags: List of extracted tags
                - ocr: Text found in the image
                - objects: Detected objects with bounding boxes
                - style: Style description
                - color: Dominant color
        """
        self._load_model()

        try:
            # Generate detailed caption
            caption = self._generate_caption(image, context)

            # Extract tags from caption
            tags = self._extract_tags(caption)

            # Perform OCR
            ocr_text = self._extract_text(image)

            # Detect objects (optional, can be expensive)
            objects = []  # self._detect_objects(image) if needed

            # Analyze style and color
            style = self._analyze_style(caption)
            color = self._analyze_color(image)

            result = {
                "caption": caption,
                "tags": tags,
                "ocr": ocr_text,
                "objects": objects,
                "style": style,
                "color": color
            }

            logger.info(f"Analysis complete: {len(tags)} tags, OCR: {bool(ocr_text)}")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "caption": "",
                "tags": [],
                "ocr": "",
                "objects": [],
                "style": "",
                "color": ""
            }

    def analyze_file(self, image_path: Path, context: Optional[Dict] = None) -> Dict[str, Any]:
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
            return {
                "caption": "",
                "tags": [],
                "ocr": "",
                "objects": [],
                "style": "",
                "color": ""
            }

    def _generate_caption(
        self,
        image: Image.Image,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate detailed caption for the image.

        Args:
            image: PIL Image
            context: Optional context (layer type, name, etc.)

        Returns:
            Detailed caption string
        """
        try:
            if "blip2" in self.model_id.lower():
                # BLIP-2 caption generation
                prompt = "a detailed description of"
                if context and context.get('layer_type'):
                    layer_type = context['layer_type']
                    prompt = f"a detailed description of this {layer_type} layer:"

                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    length_penalty=1.0
                )

                caption = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()

                return caption

            elif "blip" in self.model_id.lower():
                # BLIP (original) - unconditional caption generation
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3
                )

                caption = self.processor.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )

                return caption

            elif "Qwen2-VL" in self.model_id or "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id:
                # Qwen2-VL / Qwen3-VL caption generation (same chat template API)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                }]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                caption = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                return caption
            else:
                # Florence-2 format (if ever fixed)
                prompt = "<DETAILED_CAPTION>"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=256, num_beams=3)
                result = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                return self._parse_florence_output(result)

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return ""

    def _extract_text(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR.

        Args:
            image: PIL Image

        Returns:
            Extracted text string
        """
        try:
            if "blip2" in self.model_id.lower():
                # BLIP-2 with text extraction prompt
                # Note: BLIP-2 is not optimized for OCR, consider adding EasyOCR/TrOCR later
                prompt = "Question: What text is visible in this image? Answer:"

                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=3
                )

                text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()

                # If response is generic (no text found), return empty
                if "no text" in text.lower() or "cannot" in text.lower():
                    return ""

                return text

            elif "blip" in self.model_id.lower():
                # BLIP (original) - use VQA for text extraction
                # Note: BLIP is not optimized for OCR
                text_prompt = "what text is written in this image?"

                inputs = self.processor(
                    images=image,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.model.generate(
                    **inputs,
                    max_length=100
                )

                text = self.processor.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )

                # If response is generic, return empty
                if any(word in text.lower() for word in ["no text", "cannot", "none", "unclear"]):
                    return ""

                return text

            elif "Qwen2-VL" in self.model_id:
                # Qwen2-VL OCR
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Extract all text visible in this image. If no text, respond with 'none'."}
                    ]
                }]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                result = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                return result if "none" not in result.lower() else ""
            else:
                # Florence-2 OCR format
                prompt = "<OCR>"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=3)
                result = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                return self._parse_florence_output(result)

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def _extract_tags(self, caption: str) -> List[str]:
        """
        Extract relevant tags from caption.

        Args:
            caption: Image caption text

        Returns:
            List of tags
        """
        if not caption:
            return []

        # Simple keyword extraction (can be improved with NLP)
        import re

        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'this', 'that'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', caption.lower())

        # Filter and deduplicate
        tags = []
        seen = set()

        for word in words:
            if len(word) > 2 and word not in stop_words and word not in seen:
                tags.append(word)
                seen.add(word)

        return tags[:20]  # Limit to 20 tags

    def _analyze_style(self, caption: str) -> str:
        """
        Infer style from caption.

        Args:
            caption: Image caption

        Returns:
            Style description
        """
        # Simple keyword matching for style
        style_keywords = {
            'minimalist', 'modern', 'vintage', 'retro', 'elegant',
            'playful', 'professional', 'artistic', 'cinematic',
            'dark', 'bright', 'colorful', 'monochrome'
        }

        found_styles = []
        caption_lower = caption.lower()

        for keyword in style_keywords:
            if keyword in caption_lower:
                found_styles.append(keyword)

        return ', '.join(found_styles) if found_styles else ""

    def _analyze_color(self, image: Image.Image) -> str:
        """
        Extract dominant color from image.

        Args:
            image: PIL Image

        Returns:
            Hex color code
        """
        try:
            # Resize for faster processing
            img = image.copy()
            img.thumbnail((100, 100))

            # Get dominant color using simple averaging
            import numpy as np

            pixels = np.array(img)
            avg_color = pixels.mean(axis=(0, 1)).astype(int)

            # Convert to hex
            hex_color = f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}"

            return hex_color

        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return ""

    def _parse_florence_output(self, output: str) -> str:
        """
        Parse Florence-2 model output format.

        Florence-2 returns results in a specific format that needs parsing.

        Args:
            output: Raw model output

        Returns:
            Parsed text
        """
        # Florence-2 output format: <TASK>result</TASK>
        # Extract content between tags
        import re

        # Try to extract content after task tags
        match = re.search(r'</s>(.*?)(?:</s>|$)', output)
        if match:
            return match.group(1).strip()

        # Fallback: return cleaned output
        cleaned = output.replace('<s>', '').replace('</s>', '')
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        return cleaned.strip()

    # ── v3 P0: 2-Stage Pipeline (transformers backend) ─────────

    def _generate_response(self, image: Image.Image, prompt: str) -> str:
        """
        Generate text response from Qwen2/3-VL using chat template.

        This is the core inference helper shared by classify() and analyze_structured().
        Uses the same chat template API as _generate_caption() but with arbitrary prompts
        and proper input-token trimming to return model output only.

        Args:
            image: PIL Image
            prompt: Text prompt for the model

        Returns:
            Model-generated text (input tokens stripped)
        """
        self._load_model()

        if "Qwen2-VL" in self.model_id or "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            # Trim input tokens to get model output only
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True
            )[0]
        else:
            # Fallback for non-Qwen models: use _generate_caption with the prompt
            return self._generate_caption(image, context={"prompt_override": prompt})

    def classify(self, image: Image.Image, keep_alive: str = None) -> Dict[str, Any]:
        """
        Stage 1: Classify image type.

        Args:
            image: PIL Image
            keep_alive: Ignored (Ollama-only concept, kept for interface compatibility)

        Returns:
            {"image_type": "character"|"background"|..., "confidence": "high"|...}
        """
        from .prompts import STAGE1_PROMPT
        from .schemas import STAGE1_SCHEMA
        from .repair import parse_structured_output

        try:
            t0 = time.perf_counter()
            raw = self._generate_response(image, STAGE1_PROMPT)
            elapsed = time.perf_counter() - t0
            result = parse_structured_output(raw, STAGE1_SCHEMA, image_type="other")
            logger.info(f"Stage 1 completed in {elapsed:.1f}s → {result.get('image_type')}")
            return result
        except Exception as e:
            logger.warning(f"Stage 1 classification failed: {e}")
            return {"image_type": "other", "confidence": "low"}

    def analyze_structured(
        self, image: Image.Image, image_type: str, keep_alive: str = None, context: dict = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Type-specific structured analysis.

        Args:
            image: PIL Image
            image_type: Result from Stage 1 classify()
            keep_alive: Ignored (Ollama-only concept)
            context: Optional file metadata context (v3.1: MC.raw)

        Returns:
            Structured dict with type-specific fields + caption + tags
        """
        from .prompts import get_stage2_prompt
        from .schemas import get_schema
        from .repair import parse_structured_output

        try:
            t0 = time.perf_counter()
            prompt = get_stage2_prompt(image_type, context=context)
            raw = self._generate_response(image, prompt)
            elapsed = time.perf_counter() - t0
            result = parse_structured_output(raw, get_schema(image_type), image_type=image_type)
            result["image_type"] = image_type
            logger.info(f"Stage 2 completed in {elapsed:.1f}s ({image_type})")
            return result
        except Exception as e:
            logger.warning(f"Stage 2 analysis failed: {e}")
            return {"caption": "", "tags": [], "image_type": image_type}

    def classify_and_analyze(
        self, image: Image.Image, keep_alive: str = None, context: dict = None
    ) -> Dict[str, Any]:
        """
        Full 2-Stage pipeline: classify → analyze_structured.

        Args:
            image: PIL Image to analyze
            keep_alive: Ignored (Ollama-only concept)
            context: Optional file metadata context (v3.1: MC.raw)

        Returns:
            Merged dict with image_type + all structured fields
        """
        t_total = time.perf_counter()

        classification = self.classify(image, keep_alive)
        image_type = classification.get("image_type", "other")
        logger.info(f"Stage 1 → {image_type} (confidence: {classification.get('confidence', '?')})")

        analysis = self.analyze_structured(image, image_type, keep_alive, context=context)

        total_elapsed = time.perf_counter() - t_total
        logger.info(f"2-Stage total: {total_elapsed:.1f}s ({image_type})")

        return analysis

    # ── Batch inference helpers ─────────────────────────────────

    def _is_qwen_vl(self) -> bool:
        return "Qwen2-VL" in self.model_id or "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id

    def _generate_response_batch(
        self, images: List[Image.Image], prompts
    ) -> List[str]:
        """
        Batch generate: multiple images with prompts → list of responses.

        Args:
            images: List of PIL Images
            prompts: Single prompt string (same for all) or list of per-image prompts

        Falls back to sequential if not Qwen VL.
        """
        # Normalize prompts to list
        if isinstance(prompts, str):
            prompt_list = [prompts] * len(images)
        else:
            prompt_list = prompts

        if not self._is_qwen_vl() or len(images) <= 1:
            return [self._generate_response(img, p) for img, p in zip(images, prompt_list)]

        self._load_model()
        texts = []
        for img, prompt in zip(images, prompt_list):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)

    def classify_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Stage 1 batch: classify multiple images at once."""
        from .prompts import STAGE1_PROMPT
        from .schemas import STAGE1_SCHEMA
        from .repair import parse_structured_output

        t0 = time.perf_counter()
        try:
            raw_list = self._generate_response_batch(images, STAGE1_PROMPT)
        except Exception as e:
            logger.warning(f"Stage 1 batch failed ({e}), falling back to sequential")
            return [self.classify(img) for img in images]

        results = []
        for raw in raw_list:
            results.append(parse_structured_output(raw, STAGE1_SCHEMA, image_type="other"))

        elapsed = time.perf_counter() - t0
        logger.info(f"Stage 1 batch: {len(images)} images in {elapsed:.1f}s")
        return results

    def analyze_structured_batch(
        self, images: List[Image.Image], image_type: str,
        contexts: List[dict] = None
    ) -> List[Dict[str, Any]]:
        """Stage 2 batch: same image_type, multiple images."""
        from .prompts import get_stage2_prompt
        from .schemas import get_schema
        from .repair import parse_structured_output

        # Build per-image prompts (same template, different context)
        # If contexts differ, we must go sequential per unique prompt
        prompts = []
        for i, img in enumerate(images):
            ctx = contexts[i] if contexts else None
            prompts.append(get_stage2_prompt(image_type, context=ctx))

        # Batch with per-image prompts (different contexts are fine)
        t0 = time.perf_counter()
        try:
            raw_list = self._generate_response_batch(images, prompts)
        except Exception as e:
            logger.warning(f"Stage 2 batch failed ({e}), falling back to sequential")
            return [
                self.analyze_structured(img, image_type, context=contexts[i] if contexts else None)
                for i, img in enumerate(images)
            ]
        elapsed = time.perf_counter() - t0
        logger.info(f"Stage 2 batch ({image_type}): {len(images)} images in {elapsed:.1f}s")

        schema = get_schema(image_type)
        results = []
        for raw in raw_list:
            result = parse_structured_output(raw, schema, image_type=image_type)
            result["image_type"] = image_type
            results.append(result)
        return results

    def classify_and_analyze_batch(
        self, items: list, vlm_batch: int = 8, progress_callback=None
    ) -> list:
        """
        Full 2-Stage batch pipeline: batch classify → group by type → batch analyze.

        Args:
            items: List of (image, context) tuples
            vlm_batch: Max images per GPU batch (VRAM limit)
            progress_callback: fn(idx, total, result)
        """
        self._load_model()
        n = len(items)
        images = [img for img, _ in items]
        contexts = [ctx for _, ctx in items]
        results = [None] * n

        # ── Stage 1: Batch classify (sub-batch for VRAM safety) ──
        classifications = []
        for sb_start in range(0, n, vlm_batch):
            sb_images = images[sb_start:sb_start + vlm_batch]
            classifications.extend(self.classify_batch(sb_images))

        # ── Group by image_type for Stage 2 ──
        type_groups = {}  # image_type → [(original_idx, image, context)]
        for idx, cls in enumerate(classifications):
            it = cls.get("image_type", "other")
            type_groups.setdefault(it, []).append((idx, images[idx], contexts[idx]))

        logger.info(
            f"Stage 1 done: {n} images → "
            + ", ".join(f"{t}:{len(g)}" for t, g in type_groups.items())
        )

        # ── Stage 2: Batch per type (sub-batch for VRAM safety) ──
        progress_idx = 0
        for image_type, group in type_groups.items():
            group_images = [g[1] for g in group]
            group_contexts = [g[2] for g in group]
            group_indices = [g[0] for g in group]

            group_results = []
            for sb_start in range(0, len(group), vlm_batch):
                sb_imgs = group_images[sb_start:sb_start + vlm_batch]
                sb_ctxs = group_contexts[sb_start:sb_start + vlm_batch]
                sb_results = self.analyze_structured_batch(
                    sb_imgs, image_type, contexts=sb_ctxs
                )
                group_results.extend(sb_results)

            for i, result in enumerate(group_results):
                orig_idx = group_indices[i]
                results[orig_idx] = result
                if progress_callback:
                    progress_callback(progress_idx, n, result)
                progress_idx += 1

        # Fill any None with fallback
        for i in range(n):
            if results[i] is None:
                results[i] = {"caption": "", "tags": [], "image_type": "other"}

        return results

    def classify_and_analyze_sequence(
        self, items: list, progress_callback=None
    ) -> list:
        """Process multiple images — batch if supported, else sequential."""
        self._load_model()

        # Use batch path for Qwen VL models
        if self._is_qwen_vl() and len(items) > 1:
            try:
                from backend.utils.tier_config import get_active_tier
                _, tier_cfg = get_active_tier()
                vlm_batch = tier_cfg.get("vlm", {}).get("batch_size", 8)
                logger.info(f"VLM batch mode: {len(items)} images, batch_size={vlm_batch}")
                return self.classify_and_analyze_batch(
                    items, vlm_batch=vlm_batch, progress_callback=progress_callback
                )
            except Exception as e:
                logger.warning(f"Batch VLM failed ({e}), falling back to sequential")

        # Sequential fallback
        results = []
        for idx, (image, context) in enumerate(items):
            try:
                result = self.classify_and_analyze(image, context=context)
            except Exception as e:
                logger.warning(f"Vision failed for item {idx}: {e}")
                result = {"caption": "", "tags": [], "image_type": "other"}
            results.append(result)
            if progress_callback:
                progress_callback(idx, len(items), result)
        return results
