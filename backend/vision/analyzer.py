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
import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from backend.utils.config import get_config

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
        cfg = get_config()
        self.require_local_models = bool(cfg.get("vision.require_local_models", True))
        self.fail_if_cpu_on_macos = bool(cfg.get("vision.fail_if_cpu_on_macos", True))
        default_max_new_tokens = 96 if self.device == "cpu" else 192
        default_max_gen_time_s = 12.0 if self.device == "cpu" else 30.0
        self.vlm_max_new_tokens = int(
            cfg.get("vision.vlm_max_new_tokens", default_max_new_tokens) or default_max_new_tokens
        )
        self.vlm_max_gen_time_s = float(
            cfg.get("vision.vlm_max_gen_time_s", default_max_gen_time_s) or default_max_gen_time_s
        )
        if self.require_local_models:
            # Force strict offline mode to avoid long network retries when local models are missing.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(
            f"VisionAnalyzer initialized (tier: {tier_name}, device: {self.device}, "
            f"model: {self.model_id}, dtype: {dtype}, "
            f"max_new_tokens: {self.vlm_max_new_tokens}, max_time_s: {self.vlm_max_gen_time_s})"
        )
        if platform.system() == "Darwin" and self.device == "cpu":
            mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            msg = (
                "macOS detected but VLM device resolved to CPU "
                f"(mps_built={mps_built}, mps_available={mps_available}). "
                "This will cause severe stalls."
            )
            if self.fail_if_cpu_on_macos:
                raise RuntimeError(
                    f"{msg} Set up a working MPS runtime or disable "
                    "`vision.fail_if_cpu_on_macos` only for emergency fallback."
                )
            logger.warning(msg)

    def _generate_with_limits(self, **inputs):
        """Generate with conservative defaults to prevent long stalls on CPU."""
        kwargs = {"max_new_tokens": self.vlm_max_new_tokens}
        if self.vlm_max_gen_time_s > 0:
            kwargs["max_time"] = self.vlm_max_gen_time_s
        try:
            return self.model.generate(**inputs, **kwargs)
        except TypeError:
            kwargs.pop("max_time", None)
            return self.model.generate(**inputs, **kwargs)

    def _configure_padding_for_decoder_generation(self):
        """
        Configure tokenizer padding for decoder-only generation models.

        Qwen-VL family expects left padding in batched generation.
        """
        tok = None
        if self.processor is None:
            return
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            tok = self.processor.tokenizer
        elif hasattr(self.processor, "padding_side"):
            tok = self.processor

        if tok is None:
            return

        try:
            if getattr(tok, "padding_side", None) != "left":
                tok.padding_side = "left"
                logger.info("[VLM load] tokenizer padding_side set to left")
            if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
                logger.info("[VLM load] tokenizer pad_token set to eos_token")
        except Exception as e:
            logger.warning(f"[VLM load] tokenizer padding config skipped: {e}")

    def _load_model(self):
        """Lazy load the vision model."""
        if self.model is not None:
            return

        logger.info(f"Loading {self.model_id} model...")
        hf_kwargs = {"local_files_only": self.require_local_models}

        try:
            if "blip2" in self.model_id.lower():
                # BLIP-2 model
                from transformers import Blip2Processor, Blip2ForConditionalGeneration

                self.processor = Blip2Processor.from_pretrained(self.model_id, **hf_kwargs)
                dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
                blip_kwargs = {"torch_dtype": dtype, **hf_kwargs}
                if self.device in ("cuda", "mps"):
                    blip_kwargs["device_map"] = {"": self.device}
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id, **blip_kwargs,
                )
                if self.device not in ("cuda", "mps"):
                    self.model = self.model.to(self.device)

            elif "blip" in self.model_id.lower():
                # BLIP (original) model (default, simple and reliable)
                from transformers import BlipProcessor, BlipForConditionalGeneration

                self.processor = BlipProcessor.from_pretrained(self.model_id, **hf_kwargs)
                dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
                blip_kwargs = {"torch_dtype": dtype, **hf_kwargs}
                if self.device in ("cuda", "mps"):
                    blip_kwargs["device_map"] = {"": self.device}
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_id, **blip_kwargs,
                )
                if self.device not in ("cuda", "mps"):
                    self.model = self.model.to(self.device)

            elif "Qwen2-VL" in self.model_id:
                # Qwen2-VL model
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(self.model_id, **hf_kwargs)
                self._configure_padding_for_decoder_generation()
                qwen2_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
                qwen2_dm = {"": self.device} if self.device in ("cuda", "mps") else "auto"
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=qwen2_dtype,
                    device_map=qwen2_dm,
                    **hf_kwargs,
                )

            elif "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id:
                # Qwen3-VL model (Standard/Pro/Ultra tiers)
                # Use Auto classes for forward-compatible HF pattern
                from transformers import AutoProcessor, AutoModelForImageTextToText

                t_proc = time.perf_counter()
                logger.info("[VLM load] AutoProcessor.from_pretrained start")
                self.processor = AutoProcessor.from_pretrained(self.model_id, **hf_kwargs)
                self._configure_padding_for_decoder_generation()
                logger.info(
                    f"[VLM load] AutoProcessor ready in {time.perf_counter() - t_proc:.1f}s"
                )

                # Load directly onto target device via device_map to avoid
                # 2x memory peak from CPU staging (MPS supported since transformers 5.0+)
                t_model = time.perf_counter()
                if self.device in ("cuda", "mps"):
                    dm = {"": self.device}
                    logger.info(f"[VLM load] from_pretrained start (device_map={{\"\":\"{self.device}\"}})")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        device_map=dm,
                        **hf_kwargs,
                    )
                else:
                    logger.info("[VLM load] from_pretrained start (cpu path)")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        **hf_kwargs,
                    )
                logger.info(
                    f"[VLM load] Weights loaded in {time.perf_counter() - t_model:.1f}s"
                )

            else:
                # Florence-2 or other models
                from transformers import AutoProcessor, AutoModelForCausalLM

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    **hf_kwargs,
                )
                other_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
                other_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": other_dtype,
                    **hf_kwargs,
                }
                if self.device in ("cuda", "mps"):
                    other_kwargs["device_map"] = {"": self.device}
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **other_kwargs,
                )
                if self.device not in ("cuda", "mps"):
                    self.model = self.model.to(self.device)

            logger.info(f"{self.model_id} loaded successfully on {self.device}")

        except Exception as e:
            # Do not mask ingest-engine timeout interrupts as "model missing".
            # _call_with_timeout raises a private _TimedOut inside the target call.
            if isinstance(e, TimeoutError) or e.__class__.__name__ == "_TimedOut":
                raise
            if self.require_local_models:
                msg = (
                    f"Vision model not installed locally: {self.model_id}. "
                    "Run: .venv/bin/python scripts/install_hf_models.py --vlm"
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
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

                # Free GPU tensors + clear rope_deltas cache
                del inputs, generated_ids
                if hasattr(self.model, 'rope_deltas'):
                    self.model.rope_deltas = None

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

    def _generate_response(self, image: Image.Image, user_prompt: str, system_prompt: str = None) -> str:
        """
        Generate text response from Qwen2/3-VL using chat template.

        This is the core inference helper shared by classify() and analyze_structured().
        Uses HF recommended 1-step apply_chat_template(tokenize=True) pattern.

        Args:
            image: PIL Image
            user_prompt: Text prompt for the model (user role)
            system_prompt: Optional system role message for JSON enforcement

        Returns:
            Model-generated text (input tokens stripped)
        """
        self._load_model()

        if "Qwen2-VL" in self.model_id or "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            })

            # HF recommended 1-step apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self._generate_with_limits(**inputs)

            # Trim input tokens to get model output only
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], generated_ids)
            ]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True
            )[0]

            # Free GPU tensors + CPU intermediates (MPS doesn't auto-reclaim)
            del inputs, generated_ids, trimmed
            # Clear model's cached rope_deltas to prevent memory accumulation
            if hasattr(self.model, 'rope_deltas'):
                self.model.rope_deltas = None
            import gc; gc.collect()
            if self.device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()

            return decoded
        else:
            # Fallback for non-Qwen models: use _generate_caption with the prompt
            return self._generate_caption(image, context={"prompt_override": user_prompt})

    def classify(self, image: Image.Image, keep_alive: str = None, domain=None) -> Dict[str, Any]:
        """
        Stage 1: Classify image type.

        Args:
            image: PIL Image
            keep_alive: Ignored (Ollama-only concept, kept for interface compatibility)
            domain: Optional DomainProfile for domain-scoped classification

        Returns:
            {"image_type": "character"|"background"|..., "confidence": "high"|...}
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
            logger.info(f"Stage 1 completed in {elapsed:.1f}s → {result.get('image_type')}")
            return result
        except Exception as e:
            logger.warning(f"Stage 1 classification failed: {e}")
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
            keep_alive: Ignored (Ollama-only concept)
            context: Optional file metadata context (v3.1: MC.raw)
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
            result = parse_structured_output(raw, get_schema(image_type), image_type=image_type)
            result["image_type"] = image_type
            logger.info(f"Stage 2 completed in {elapsed:.1f}s ({image_type})")
            return result
        except Exception as e:
            logger.warning(f"Stage 2 analysis failed: {e}")
            return {"caption": "", "tags": [], "image_type": image_type}

    def classify_and_analyze(
        self, image: Image.Image, keep_alive: str = None, context: dict = None,
        domain=None
    ) -> Dict[str, Any]:
        """
        Full 2-Stage pipeline: classify → analyze_structured.

        Args:
            image: PIL Image to analyze
            keep_alive: Ignored (Ollama-only concept)
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

    # ── Batch inference helpers ─────────────────────────────────

    def _is_qwen_vl(self) -> bool:
        return "Qwen2-VL" in self.model_id or "Qwen3-VL" in self.model_id or "Qwen3VL" in self.model_id

    def _generate_response_batch(
        self, images: List[Image.Image], prompts, system_prompt: str = None
    ) -> List[str]:
        """
        Batch generate: multiple images with prompts → list of responses.

        Args:
            images: List of PIL Images
            prompts: Single prompt string (same for all) or list of per-image prompts
            system_prompt: Optional system role message (same for all images)

        Falls back to sequential if not Qwen VL.
        """
        # Normalize prompts to list
        if isinstance(prompts, str):
            prompt_list = [prompts] * len(images)
        else:
            prompt_list = prompts

        if not self._is_qwen_vl() or len(images) <= 1:
            return [self._generate_response(img, p, system_prompt) for img, p in zip(images, prompt_list)]

        self._load_model()

        # Batch apply_chat_template: per-item since tokenize=True returns tensors
        # that can't be batched directly across variable-length sequences.
        # Use 2-step for batch: apply_chat_template(tokenize=False) + processor()
        texts = []
        for img, prompt in zip(images, prompt_list):
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            })
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._generate_with_limits(**inputs)

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)

        # Free GPU tensors + CPU intermediates (MPS doesn't auto-reclaim)
        del inputs, generated_ids, trimmed
        # Clear model's cached rope_deltas to prevent memory accumulation
        if hasattr(self.model, 'rope_deltas'):
            self.model.rope_deltas = None
        import gc; gc.collect()
        if self.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return decoded

    def classify_batch(self, images: List[Image.Image], domain=None) -> List[Dict[str, Any]]:
        """Stage 1 batch: classify multiple images at once."""
        from .prompts import STAGE1_USER, STAGE1_SYSTEM, build_stage1_prompt
        from .schemas import STAGE1_SCHEMA
        from .repair import parse_structured_output

        prompt = build_stage1_prompt(domain) if domain else STAGE1_USER
        t0 = time.perf_counter()
        try:
            raw_list = self._generate_response_batch(images, prompt, STAGE1_SYSTEM)
        except Exception as e:
            logger.warning(f"Stage 1 batch failed ({e}), falling back to sequential")
            return [self.classify(img, domain=domain) for img in images]

        results = []
        for raw in raw_list:
            results.append(parse_structured_output(raw, STAGE1_SCHEMA, image_type="other"))

        elapsed = time.perf_counter() - t0
        logger.info(f"Stage 1 batch: {len(images)} images in {elapsed:.1f}s")
        return results

    def analyze_structured_batch(
        self, images: List[Image.Image], image_type: str,
        contexts: List[dict] = None, domain=None
    ) -> List[Dict[str, Any]]:
        """Stage 2 batch: same image_type, multiple images."""
        from .prompts import get_stage2_prompt, STAGE2_SYSTEM
        from .schemas import get_schema
        from .repair import parse_structured_output

        # Build per-image prompts (same template, different context)
        # If contexts differ, we must go sequential per unique prompt
        prompts = []
        for i, img in enumerate(images):
            ctx = contexts[i] if contexts else None
            prompts.append(get_stage2_prompt(image_type, context=ctx, domain=domain))

        # Batch with per-image prompts (different contexts are fine)
        t0 = time.perf_counter()
        try:
            raw_list = self._generate_response_batch(images, prompts, STAGE2_SYSTEM)
        except Exception as e:
            logger.warning(f"Stage 2 batch failed ({e}), falling back to sequential")
            return [
                self.analyze_structured(img, image_type, context=contexts[i] if contexts else None, domain=domain)
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
        self, items: list, vlm_batch: int = 8, progress_callback=None, domain=None
    ) -> list:
        """
        Full 2-Stage batch pipeline: batch classify → group by type → batch analyze.

        Args:
            items: List of (image, context) tuples
            vlm_batch: Max images per GPU batch (VRAM limit)
            progress_callback: fn(idx, total, result)
            domain: Optional DomainProfile for domain-aware classification
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
            classifications.extend(self.classify_batch(sb_images, domain=domain))

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
                    sb_imgs, image_type, contexts=sb_ctxs, domain=domain
                )
                group_results.extend(sb_results)

            for i, result in enumerate(group_results):
                orig_idx = group_indices[i]
                results[orig_idx] = result
                if progress_callback:
                    progress_callback(progress_idx, n, result)
                progress_idx += 1

        # Release intermediate image references (PIL Images owned by caller)
        del images, contexts, classifications, type_groups

        # Fill any None with fallback
        for i in range(n):
            if results[i] is None:
                results[i] = {"caption": "", "tags": [], "image_type": "other"}

        return results

    def classify_and_analyze_sequence(
        self, items: list, progress_callback=None, domain=None
    ) -> list:
        """Process multiple images — batch if supported, else sequential."""
        self._load_model()

        # Use batch path for Qwen VL models, except CPU where long stalls are common.
        if self._is_qwen_vl() and len(items) > 1 and self.device != "cpu":
            try:
                from backend.utils.tier_config import get_active_tier
                _, tier_cfg = get_active_tier()
                vlm_batch = tier_cfg.get("vlm", {}).get("batch_size", 8)
                logger.info(f"VLM batch mode: {len(items)} images, batch_size={vlm_batch}")
                return self.classify_and_analyze_batch(
                    items, vlm_batch=vlm_batch, progress_callback=progress_callback,
                    domain=domain
                )
            except Exception as e:
                logger.warning(f"Batch VLM failed ({e}), falling back to sequential")
        elif self._is_qwen_vl() and len(items) > 1 and self.device == "cpu":
            logger.info("VLM batch disabled on CPU device; using sequential mode for stable progress")

        # Sequential fallback
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
        """Unload VLM model and processor to free GPU memory."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info(f"VisionAnalyzer model unloaded ({self.model_id})")
