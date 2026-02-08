"""
LLM Query Decomposer - Converts natural language queries to structured search parameters.

Uses Ollama Chat API (text-only, no images) to decompose queries into:
- vector_query: English text for CLIP vector search
- fts_keywords: Keywords for FTS5 full-text search (Korean + English)
- filters: Structured metadata filters (format, color, etc.)
- query_type: Query classification for auto-weighted RRF

Uses assistant prefix with empty <think> block to suppress Qwen3 thinking,
reducing response time from ~100s to ~3s.

Falls back gracefully when Ollama is unavailable.
"""

import logging
import json
import os
from typing import Dict, Any

import requests

logger = logging.getLogger(__name__)

# Default from .env
_DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:8b")


class QueryDecomposer:
    """Decomposes natural language search queries using LLM."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
    ):
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/chat"
        logger.info(f"QueryDecomposer initialized (model: {model}, host: {host})")

    def decompose(self, query: str) -> Dict[str, Any]:
        """
        Decompose a natural language query into structured search parameters.

        Args:
            query: User's natural language search query

        Returns:
            {
                "vector_query": str,      # English CLIP search query
                "fts_keywords": list,     # FTS5 keywords (mixed lang)
                "filters": dict,          # Metadata filters
                "query_type": str,        # visual/keyword/semantic/balanced
                "decomposed": bool        # True if LLM was used
            }
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": self._build_prompt(query)},
                        {"role": "assistant", "content": "<think>\n</think>\n{"},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 256},
                    "keep_alive": "5m",
                },
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning(f"Ollama returned status {response.status_code}")
                return self._fallback(query)

            result = response.json()
            msg = result.get("message", {})
            raw_text = "{" + msg.get("content", "")
            parsed = self._parse_response(raw_text, query)
            parsed["decomposed"] = True
            logger.info(f"Query decomposed: '{query}' → vector='{parsed['vector_query']}', "
                        f"fts={parsed['fts_keywords']}, filters={parsed['filters']}, "
                        f"query_type={parsed['query_type']}")
            return parsed

        except requests.exceptions.ConnectionError:
            logger.info("Ollama not running, using fallback")
            return self._fallback(query)
        except requests.exceptions.Timeout:
            logger.warning("Ollama timeout (>30s), using fallback")
            return self._fallback(query)
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using fallback")
            return self._fallback(query)

    def _build_prompt(self, query: str) -> str:
        return f"""You are a search query decomposer for an image asset database.
Convert the user's natural language search query into structured search parameters.

The database has 16 FTS columns across two axes:

AI Vision (Qwen/Ollama output):
- ai_caption: English description of the image
- ai_tags: keywords independently generated (e.g. "anime style", "night scene")
- ai_style: style description (e.g. "anime illustration", "photorealistic")
- dominant_color: color analysis (e.g. "blue", "warm orange")
- ocr_text: text visible inside the image (signs, UI text, etc.)
- image_type: classified type (character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other)
- scene_type: background scene (alley, forest, dungeon, castle, village, etc.)
- art_style: art style (realistic, anime, pixel, painterly, cartoon, 3d_render, etc.)

Structural (file parsing + user input):
- file_path, file_name: file identity and folder path
- layer_names: PSD layer names in original + Korean + English
- text_content: text extracted from PSD text layers (original + KR + EN)
- used_fonts: font names used in PSD (e.g. "NotoSans", "Pretendard")
- user_note: user's personal memo
- user_tags: user-assigned tags
- folder_tags: folder name tags

Rules:
1. vector_query MUST be in English (for CLIP model compatibility)
2. fts_keywords should include BOTH the original language terms AND English translations
3. Include style/color/font terms when relevant (they match ai_style, dominant_color, used_fonts)
4. filters should only include fields that are clearly specified
5. Keep vector_query concise but descriptive (good for CLIP similarity)
6. If the query mentions a specific image_type, art_style, or scene_type, add it to filters

Also classify the query_type:
- "visual": query focuses on visual style, color, mood, tone, artistic feeling (e.g. "파란 톤", "warm mood illustration")
- "keyword": query contains specific named objects, scene types, or entities (e.g. "dragon", "castle", "야경")
- "semantic": query describes purpose, context, usage, or abstract concept (e.g. "할인 이벤트 배너", "login screen UI")
- "balanced": mixed or unclear

User query: "{query}"

Return ONLY valid JSON (no markdown, no explanation):
{{"vector_query": "english description for CLIP search", "fts_keywords": ["keyword1", "keyword2"], "filters": {{}}, "query_type": "visual|keyword|semantic|balanced"}}

Supported filter keys: "format" (PSD/PNG/JPG), "dominant_color_hint" (color name), "image_type", "art_style", "scene_type", "time_of_day", "weather\""""

    def _parse_response(self, text: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])

                vector_query = data.get("vector_query", original_query)
                fts_keywords = data.get("fts_keywords", [original_query])
                filters = data.get("filters", {})

                # Ensure fts_keywords is a list
                if isinstance(fts_keywords, str):
                    fts_keywords = [fts_keywords]

                # Always include original query in fts_keywords for safety
                if original_query not in fts_keywords:
                    fts_keywords.append(original_query)

                query_type = data.get("query_type", "balanced")
                if query_type not in ("visual", "keyword", "semantic", "balanced"):
                    query_type = "balanced"

                return {
                    "vector_query": vector_query if vector_query else original_query,
                    "fts_keywords": fts_keywords,
                    "filters": filters if isinstance(filters, dict) else {},
                    "query_type": query_type,
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return self._fallback(original_query)

    def _fallback(self, query: str) -> Dict[str, Any]:
        """Fallback when LLM is unavailable - translate query for cross-language search."""
        vector_query = query
        fts_keywords = [query]

        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='auto', target='en').translate(query)
            if translated and translated.lower() != query.lower():
                vector_query = translated
                fts_keywords.append(translated)
                logger.info(f"Fallback translated: '{query}' → '{translated}'")
        except Exception as e:
            logger.debug(f"Fallback translation skipped: {e}")

        return {
            "vector_query": vector_query,
            "fts_keywords": fts_keywords,
            "filters": {},
            "query_type": "balanced",
            "decomposed": False,
        }
