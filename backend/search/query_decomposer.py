"""
LLM Query Decomposer - Converts natural language queries to structured search parameters.

Uses Ollama Chat API (text-only, no images) to decompose queries into:
- vector_query: English text for SigLIP2 vector search (positive only)
- negative_query: English text describing things to exclude (vector penalty)
- fts_keywords: Keywords for FTS5 full-text search (Korean + English)
- exclude_keywords: Keywords to exclude from FTS5 results (Korean + English)
- filters: Structured metadata filters (format, color, etc.)
- query_type: Query classification for auto-weighted RRF

Handles negation expressions (Korean/English):
- Korean: ~없어야, ~없는, ~아닌, ~말고, ~빼고, ~제외하고, ~없이
- English: without, no ~, not ~, except, exclude

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
                "vector_query": str,       # English SigLIP2 search query (positive only)
                "negative_query": str,     # English description of things to exclude
                "fts_keywords": list,      # FTS5 keywords (mixed lang)
                "exclude_keywords": list,  # Keywords to exclude from FTS5
                "filters": dict,           # Metadata filters
                "query_type": str,         # visual/keyword/semantic/balanced
                "decomposed": bool         # True if LLM was used
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
                        f"negative='{parsed.get('negative_query', '')}', "
                        f"fts={parsed['fts_keywords']}, "
                        f"exclude={parsed.get('exclude_keywords', [])}, "
                        f"filters={parsed['filters']}, "
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
- mc_caption: English description of the image
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
1. vector_query MUST be in English (for SigLIP2 model compatibility)
2. fts_keywords should include BOTH the original language terms AND English translations
3. Include style/color/font terms when relevant (they match ai_style, dominant_color, used_fonts)
4. filters should only include fields that are clearly specified
5. Keep vector_query concise but descriptive (good for SigLIP2 similarity)
6. If the query mentions a specific image_type, art_style, or scene_type, add it to filters

Negation handling (CRITICAL):
7. Detect negation expressions in the query:
   - Korean: ~없어야, ~없는, ~아닌, ~말고, ~빼고, ~제외하고, ~없이, ~안 되는, ~하지 않은
   - English: without, no ~, not ~, except, exclude, excluding, other than
8. When negation is detected:
   - vector_query: include ONLY positive/desired elements (REMOVE all negated concepts)
   - negative_query: describe the negated/excluded visual concepts in English
   - fts_keywords: include ONLY positive search terms
   - exclude_keywords: list the negated terms in both Korean and English
9. Examples:
   - "야간 골목길 배경, 추운 느낌이 없어야 한다"
     → vector_query: "dark alley at night"
     → negative_query: "cold winter snowy icy"
     → fts_keywords: ["야간", "골목길", "배경", "night", "alley", "dark alley"]
     → exclude_keywords: ["추운", "cold", "winter", "snow"]
   - "사람이 없는 숲 풍경"
     → vector_query: "empty forest landscape without people"
     → negative_query: "person human people crowd"
     → fts_keywords: ["숲", "풍경", "forest", "landscape"]
     → exclude_keywords: ["사람", "person", "people", "human"]
   - "빨간색 제외한 꽃 일러스트"
     → vector_query: "flower illustration"
     → negative_query: "red crimson scarlet"
     → fts_keywords: ["꽃", "일러스트", "flower", "illustration"]
     → exclude_keywords: ["빨간색", "빨간", "red"]
10. If no negation is found, set negative_query to "" and exclude_keywords to []

Also classify the query_type:
- "visual": query focuses on visual style, color, mood, tone, artistic feeling (e.g. "파란 톤", "warm mood illustration")
- "keyword": query contains specific named objects, scene types, or entities (e.g. "dragon", "castle", "야경")
- "semantic": query describes purpose, context, usage, or abstract concept (e.g. "할인 이벤트 배너", "login screen UI")
- "balanced": mixed or unclear

User query: "{query}"

Return ONLY valid JSON (no markdown, no explanation):
{{"vector_query": "positive english description only", "negative_query": "english terms to exclude", "fts_keywords": ["positive", "keywords"], "exclude_keywords": ["negative", "keywords"], "filters": {{}}, "query_type": "visual|keyword|semantic|balanced"}}

Supported filter keys: "format" (PSD/PNG/JPG), "dominant_color_hint" (color name), "image_type", "art_style", "scene_type", "time_of_day", "weather\""""

    def _parse_response(self, text: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON including negation fields."""
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

                # Parse negation fields
                negative_query = data.get("negative_query", "")
                if not isinstance(negative_query, str):
                    negative_query = str(negative_query) if negative_query else ""

                exclude_keywords = data.get("exclude_keywords", [])
                # Ensure exclude_keywords is a list
                if isinstance(exclude_keywords, str):
                    exclude_keywords = [k.strip() for k in exclude_keywords.split(",") if k.strip()]
                elif not isinstance(exclude_keywords, list):
                    exclude_keywords = []

                return {
                    "vector_query": vector_query if vector_query else original_query,
                    "negative_query": negative_query,
                    "fts_keywords": fts_keywords,
                    "exclude_keywords": exclude_keywords,
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
            "negative_query": "",
            "fts_keywords": fts_keywords,
            "exclude_keywords": [],
            "filters": {},
            "query_type": "balanced",
            "decomposed": False,
        }
