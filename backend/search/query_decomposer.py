"""
LLM Query Decomposer - Converts natural language queries to structured search parameters.

Uses platform-optimal LLM backend to decompose queries into:
- vector_query: English text for SigLIP2 vector search (positive only)
- negative_query: English text describing things to exclude (vector penalty)
- fts_keywords: Keywords for FTS5 full-text search (Korean + English)
- exclude_keywords: Keywords to exclude from FTS5 results (Korean + English)
- filters: Structured metadata filters (format, color, etc.)
- query_type: Query classification for auto-weighted RRF

Backend resolution order:
  macOS (Apple Silicon): mlx-lm → Ollama → fallback
  Windows/Linux:         Ollama → transformers → fallback

Falls back gracefully when no LLM backend is available.
"""

import logging
import json
import os
import platform
import re
import time
from typing import Dict, Any, List, Tuple, Optional

import requests

logger = logging.getLogger(__name__)

# Default from .env
_DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("VISION_MODEL", "qwen3.5:9b")

# MLX model for query decomposition (text-only, Korean-capable)
_MLX_MODEL_ID = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

# ── Singleton LLM backend (shared across QueryDecomposer instances) ──
_llm_backend: Optional[str] = None  # "mlx" | "ollama" | None
_mlx_model = None
_mlx_tokenizer = None
_llm_initialized = False


def _init_llm_backend():
    """Initialize the best available LLM backend (once)."""
    global _llm_backend, _mlx_model, _mlx_tokenizer, _llm_initialized
    if _llm_initialized:
        return
    _llm_initialized = True

    # 0. Codex CLI (best accuracy — GPT-5.3-codex)
    import shutil
    if shutil.which("codex"):
        _llm_backend = "codex"
        logger.info("QueryDecomposer LLM backend: Codex CLI")
        return

    # macOS: try mlx-lm first
    if platform.system() == "Darwin":
        try:
            from mlx_lm import load
            logger.info(f"Loading MLX LLM for query decomposition: {_MLX_MODEL_ID}")
            _mlx_model, _mlx_tokenizer = load(_MLX_MODEL_ID)
            _llm_backend = "mlx"
            logger.info("QueryDecomposer LLM backend: mlx-lm")
            return
        except Exception as e:
            logger.info(f"MLX LLM not available: {e}")

    # All platforms: try Ollama
    try:
        resp = requests.get(f"{_DEFAULT_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            _llm_backend = "ollama"
            logger.info("QueryDecomposer LLM backend: Ollama")
            return
    except Exception:
        pass

    logger.warning("No LLM backend available for query decomposition — using rule-based fallback")
    _llm_backend = None


class QueryDecomposer:
    """Decomposes natural language search queries using LLM."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        use_codex: bool = True,
    ):
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/chat"
        self.use_codex = use_codex
        _init_llm_backend()
        effective = _llm_backend if (use_codex or _llm_backend != "codex") else "mlx/ollama/fallback"
        logger.info(f"QueryDecomposer initialized (backend: {_llm_backend}, use_codex: {use_codex}, effective: {effective})")

    # ── LRU cache for decomposition results ──────────────────────
    # Key: (query, use_codex) → Value: (result_dict, timestamp)
    # Avoids re-running LLM on "Load More" or filter-change re-searches.
    _decompose_cache: Dict[tuple, tuple] = {}
    _CACHE_MAX_SIZE = 50
    _CACHE_TTL_SEC = 300  # 5 minutes

    def decompose(self, query: str) -> Dict[str, Any]:
        """
        Decompose a natural language query into structured search parameters.
        Results are cached by (query, use_codex) for 5 minutes.

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
        import copy

        cache_key = (query.strip(), self.use_codex)
        cached = QueryDecomposer._decompose_cache.get(cache_key)
        if cached is not None:
            result, ts = cached
            if (time.time() - ts) < self._CACHE_TTL_SEC:
                logger.info(f"[DECOMP] cache hit for '{query}' (use_codex={self.use_codex})")
                out = copy.deepcopy(result)
                out["_decomp_backend"] = "cache"
                return out
            else:
                del QueryDecomposer._decompose_cache[cache_key]

        raw_text = self._generate_llm(query)
        # Determine which backend was actually used
        effective_backend = _llm_backend if (self.use_codex or _llm_backend != "codex") else "mlx"
        logger.warning(f"[DECOMP] query='{query}' llm_raw={repr(raw_text[:200]) if raw_text else 'None'} backend={effective_backend} (use_codex={self.use_codex})")
        if raw_text is not None:
            parsed = self._parse_response(raw_text, query)
            logger.warning(f"[DECOMP] parsed folder_filter='{parsed.get('folder_filter','')}' type='{parsed.get('query_type','')}'")
            parsed["decomposed"] = True
            result = self._finalize(parsed, query)
            result["_decomp_backend"] = effective_backend
        else:
            logger.warning(f"[DECOMP] LLM failed, using fallback")
            result = self._finalize(self._fallback(query), query)
            result["_decomp_backend"] = "fallback"

        # Store in cache (evict oldest if full)
        if len(QueryDecomposer._decompose_cache) >= self._CACHE_MAX_SIZE:
            oldest_key = next(iter(QueryDecomposer._decompose_cache))
            del QueryDecomposer._decompose_cache[oldest_key]
        QueryDecomposer._decompose_cache[cache_key] = (copy.deepcopy(result), time.time())

        return result

    # ── Phase B: structured ConstraintPlan output ─────────────────

    def decompose_plan(self, query: str):
        """Return a ConstraintPlan with one retry on schema violation.

        On both LLM-output failure and validation failure, falls back to
        the rule-based fallback rather than raising. The legacy
        decompose() method is unaffected.
        """
        from backend.search.constraint_plan import (
            ConstraintPlan,
            ConstraintPlanError,
            from_decomposer_output,
        )
        import json as _json

        def _try_parse(raw_text):
            if raw_text is None:
                raise ConstraintPlanError("LLM returned no text")
            try:
                payload = _json.loads(raw_text)
            except Exception as exc:
                raise ConstraintPlanError(f"LLM output is not JSON: {exc}") from exc
            return from_decomposer_output(payload)

        for attempt in (1, 2):
            try:
                raw = self._generate_llm(query)
                return _try_parse(raw)
            except ConstraintPlanError as exc:
                logger.warning(
                    "[DECOMP] schema attempt %d failed: %s", attempt, exc
                )

        # Both attempts failed — degrade gracefully via rule-based fallback.
        fb = self._fallback(query)
        return ConstraintPlan(
            folder="",
            elements=tuple(fb.get("fts_keywords") or [])[:5],
            negatives=tuple(fb.get("exclude_keywords") or [])[:5],
            vector_query=fb.get("vector_query") or query,
            query_type=fb.get("query_type") or "balanced",
            confidence=0.0,
        )

    def _generate_llm(self, query: str) -> Optional[str]:
        """Generate LLM response using the best available backend.

        Returns:
            JSON string starting with '{' or None if all backends fail.
        """
        prompt = self._build_prompt(query)

        # 0. Codex CLI (best accuracy) — skip if user disabled Codex
        if _llm_backend == "codex" and self.use_codex:
            try:
                return self._generate_codex(query)
            except Exception as e:
                logger.warning(f"Codex CLI failed: {e}")

        # When Codex is primary but disabled, try local backends as fallback
        try_local = (_llm_backend == "codex" and not self.use_codex) or _llm_backend in ("mlx", "ollama", None)

        if try_local:
            # 1. MLX backend (macOS)
            if _mlx_model is not None:
                try:
                    return self._generate_mlx(prompt)
                except Exception as e:
                    logger.warning(f"MLX generation failed: {e}")
            elif _llm_backend == "codex" and not self.use_codex:
                # MLX not loaded yet — try lazy init
                self._lazy_init_local_llm()
                if _mlx_model is not None:
                    try:
                        return self._generate_mlx(prompt)
                    except Exception as e:
                        logger.warning(f"MLX generation failed: {e}")

            # 2. Ollama backend
            try:
                import requests
                resp = requests.get(f"{self.host}/api/tags", timeout=3)
                if resp.status_code == 200:
                    return self._generate_ollama(prompt, query)
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}")

        return None

    @staticmethod
    def _lazy_init_local_llm():
        """Lazy-init local LLM backends when Codex is disabled at runtime."""
        global _mlx_model, _mlx_tokenizer
        if _mlx_model is not None:
            return
        if platform.system() == "Darwin":
            try:
                from mlx_lm import load
                logger.info(f"Lazy-loading MLX LLM: {_MLX_MODEL_ID}")
                _mlx_model, _mlx_tokenizer = load(_MLX_MODEL_ID)
            except Exception as e:
                logger.info(f"MLX LLM not available: {e}")

    def _generate_codex(self, query: str) -> Optional[str]:
        """Generate query decomposition using Codex CLI (GPT-5.3-codex).

        Uses file-based I/O: writes query to temp file, Codex writes result.
        Falls back to None if Codex fails or times out.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        # Write query to temp file
        query_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='_query.json', delete=False, dir='/tmp')
        json.dump({"query": query}, query_file)
        query_file.close()

        result_file = query_file.name.replace('_query.json', '_result.json')

        codex_prompt = (
            f"Read {query_file.name}. Analyze the Korean/English image search query and create a search plan. "
            "Separate the scope (which folder/project to search in) from the intent (what to find). "
            "Strip Korean particles (에서,중에,자료,폴더,이미지) from folder names. "
            f"Write result to {result_file} as a single-line JSON with this exact structure: "
            '{"pre_filter": {"folder": "folder name or empty string", "image_type": null, "format": null}, '
            '"search": {"query": "english description of what to find", "mode": "semantic"}, '
            '"fallback_keywords": ["korean and english keywords"]}'
            "\nExamples: "
            "'세일러문 중에서 강이 보이는거' → folder=세일러문, query=river scene with water visible "
            "'밤 도시 배경' → folder empty, query=night city background "
            "'마캬베리즈무 실내소품 중에서 어두운거' → folder=마캬베리즈무/실내소품, query=dark interior"
        )

        try:
            subprocess.run(
                ["codex", "exec", "--full-auto", "-s", "danger-full-access", codex_prompt],
                cwd=str(Path(__file__).resolve().parent.parent.parent),
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=45,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Codex CLI timed out (30s)")
            return None
        finally:
            try:
                os.unlink(query_file.name)
            except OSError:
                pass

        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    content = f.read().strip()
                os.unlink(result_file)
                if content.startswith('{'):
                    logger.info(f"Codex CLI result: {content[:200]}")
                    return content
            except Exception as e:
                logger.warning(f"Codex result read failed: {e}")
        return None

    def _generate_mlx(self, prompt: str) -> Optional[str]:
        """Generate using mlx-lm (Apple Silicon native)."""
        from mlx_lm import generate

        messages = [
            {"role": "user", "content": prompt},
        ]
        formatted = _mlx_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        raw = generate(
            _mlx_model, _mlx_tokenizer, prompt=formatted,
            max_tokens=512, verbose=False,
        )

        # Extract JSON from response
        raw = raw.strip()
        # Find first '{' to last '}'
        start = raw.find('{')
        end = raw.rfind('}')
        if start >= 0 and end > start:
            return raw[start:end + 1]
        return None

    def _generate_ollama(self, prompt: str, query: str) -> Optional[str]:
        """Generate using Ollama Chat API."""
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<think>\n</think>\n{"},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256},
                "keep_alive": "5m",
            },
            timeout=30,
        )
        if response.status_code != 200:
            return None
        result = response.json()
        msg = result.get("message", {})
        return "{" + msg.get("content", "")

    def _finalize(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Final validation pass applied to ALL decomposition results
        (LLM success, LLM parse failure, fallback).

        Ensures negation is always detected and cleaned regardless of path.
        Korean keyword extraction is always applied as a safety net
        (LLM may fail to decompose Korean queries properly).
        """
        result = self._validate_negation(result, original_query)
        result = self._augment_path_query(result, original_query)
        result = self._augment_ko_keywords(result, original_query)
        # Clean folder_filter: strip Korean particles/suffixes
        ff = result.get("folder_filter", "")
        if ff:
            cleaned = self._KO_PARTICLES.sub('', ff.strip())
            if cleaned and len(cleaned) >= 2:
                result["folder_filter"] = cleaned
        # Convert to unified search schema: {scope, find, exclude}
        # This is the single output format for all backends.
        pre_filter = result.get("pre_filter", {})
        folder = pre_filter.get("folder", "") if pre_filter else result.get("folder_filter", "")
        filters = result.get("filters", {})

        unified = {
            "scope": {
                "folder": folder,
                "image_type": filters.get("image_type") or (pre_filter.get("image_type") if pre_filter else None),
                "format": filters.get("format") or (pre_filter.get("format") if pre_filter else None),
            },
            "find": {
                "description": result.get("search", {}).get("query", "") if result.get("search") else result.get("vector_query", ""),
                "keywords": result.get("fallback_keywords", []) if result.get("fallback_keywords") else result.get("fts_keywords", []),
            },
            "exclude": {
                "description": result.get("negative_query", ""),
                "keywords": result.get("exclude_keywords", []),
            },
            "decomposed": result.get("decomposed", False),
            # Keep legacy fields for backward compatibility during transition
            "_legacy": result,
        }

        logger.info(
            f"Query decomposed: '{original_query}' → "
            f"scope={unified['scope']}, find='{unified['find']['description'][:60]}', "
            f"exclude='{unified['exclude']['description'][:30]}'")
        return unified

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
- "keyword": query contains ONLY specific names or identifiers with no visual/content intent (e.g. "세일러문", "hero.psd")
- "semantic": query describes purpose, context, usage, or abstract concept (e.g. "할인 이벤트 배너", "login screen UI")
- "balanced": mixed or unclear

IMPORTANT — folder_filter:
If the user scopes the query to a specific folder or project (e.g. "세일러문폴더에서 낮씬", "마캬베리즈무 중에서 실내"),
extract the folder/project name as "folder_filter" and put the REMAINING search intent in vector_query/fts_keywords.
Examples:
  "세일러문폴더 이미지 중에서 낮씬" → folder_filter: "세일러문", vector_query: "daytime outdoor scene", query_type: "visual"
  "마캬베리즈무에서 어두운 배경" → folder_filter: "마캬베리즈무", vector_query: "dark background", query_type: "visual"
  "세일러문" (no content query) → folder_filter: "", query_type: "keyword"

User query: "{query}"

Return ONLY valid JSON on a single line (no newlines, no markdown):
{{"folder_filter": "", "query_type": "visual|keyword|semantic|balanced", "vector_query": "positive english description only", "fts_keywords": ["positive", "keywords"], "negative_query": "", "exclude_keywords": [], "filters": {{}}}}

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

                folder_filter = data.get("folder_filter", "")
                if not isinstance(folder_filter, str):
                    folder_filter = ""

                # Codex may return pre_filter/search format instead of legacy
                pre_filter = data.get("pre_filter")
                search_intent = data.get("search")
                if pre_filter and isinstance(pre_filter, dict):
                    # Use pre_filter.folder if folder_filter is empty
                    pf_folder = pre_filter.get("folder", "")
                    if pf_folder and not folder_filter:
                        folder_filter = pf_folder
                if search_intent and isinstance(search_intent, dict):
                    sq = search_intent.get("query", "")
                    if sq and (not vector_query or vector_query == original_query):
                        vector_query = sq
                fallback_kw = data.get("fallback_keywords")

                result = {
                    "vector_query": vector_query if vector_query else original_query,
                    "negative_query": negative_query,
                    "fts_keywords": fts_keywords,
                    "exclude_keywords": exclude_keywords,
                    "filters": filters if isinstance(filters, dict) else {},
                    "folder_filter": folder_filter.strip(),
                    "query_type": query_type,
                }
                # Pass through Codex plan fields if present
                if pre_filter:
                    result["pre_filter"] = pre_filter
                if search_intent:
                    result["search"] = search_intent
                if fallback_kw:
                    result["fallback_keywords"] = fallback_kw
                return result
                # Note: _validate_negation is called by _finalize() in decompose()

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return self._fallback(original_query)

    # Korean negation patterns: (negated noun)(negation suffix)
    _KO_NEG_PATTERNS = [
        # "X이/가 없는", "X이/가 없어야"
        re.compile(r'(\S+?)(?:이|가|은|는)?\s*없(?:는|어야|이)', re.UNICODE),
        # "X 아닌", "X가 아닌"
        re.compile(r'(\S+?)(?:이|가|은|는)?\s*아닌', re.UNICODE),
        # "X 말고", "X 빼고", "X 제외"
        re.compile(r'(\S+?)(?:을|를)?\s*(?:말고|빼고|제외하고|제외한|제외)', re.UNICODE),
        # "X 없이"
        re.compile(r'(\S+?)(?:이|가|은|는)?\s*없이', re.UNICODE),
    ]

    # English negation words to strip from vector_query
    _EN_NEG_WORDS = re.compile(
        r'\b(without|no|not|except|excluding|other than)\b', re.IGNORECASE
    )

    def _validate_negation(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Rule-based validation: if the original query contains negation patterns
        but the LLM failed to populate negative_query/exclude_keywords, fix it.

        Also strips English negation words from vector_query (LLM sometimes
        leaves "without X" in vector_query even when negation was detected).
        """
        has_neg = result.get("negative_query", "").strip()
        has_excl = bool(result.get("exclude_keywords"))

        # Step 1: Detect Korean negation in original query
        ko_neg_nouns = self._extract_ko_negation(original_query)

        if ko_neg_nouns and not has_neg and not has_excl:
            # LLM completely missed negation — build exclude from regex
            logger.info(f"Negation validation: LLM missed negation, "
                        f"extracted Korean negative nouns: {ko_neg_nouns}")

            result["exclude_keywords"] = ko_neg_nouns

            # Try to translate Korean negative nouns to English for negative_query
            en_terms = self._translate_terms(ko_neg_nouns)
            if en_terms:
                result["negative_query"] = " ".join(en_terms)

        # Step 2: Clean vector_query — always strip negation words
        # Even if negative_query is empty (translation failed), we must still
        # remove English negation constructs like "without X" from vector_query.
        vq = result.get("vector_query", "")
        neg_q = result.get("negative_query", "")
        excl_kw = result.get("exclude_keywords", [])

        if vq and (neg_q or excl_kw or self._EN_NEG_WORDS.search(vq)):
            cleaned = self._EN_NEG_WORDS.sub("", vq).strip()

            # Remove negative subject terms from negative_query
            if neg_q:
                for term in neg_q.split():
                    pattern = re.compile(r'\b' + re.escape(term) + r'\w*\b', re.IGNORECASE)
                    cleaned = pattern.sub("", cleaned)

            # Also extract and remove English words after negation in VQ
            # Pattern: captures "without/no/not WORD(s)" constructs that may remain
            after_neg = re.findall(
                r'\b(?:without|no|not|except|excluding)\s+(\w+(?:\s+\w+)?)', vq, re.IGNORECASE
            )
            for phrase in after_neg:
                for word in phrase.split():
                    pattern = re.compile(r'\b' + re.escape(word) + r'\w*\b', re.IGNORECASE)
                    cleaned = pattern.sub("", cleaned)
                    # If translation failed, use these English terms as negative_query
                    if not neg_q and word.lower() not in ('a', 'an', 'the'):
                        neg_q = (neg_q + " " + word).strip() if neg_q else word

            cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
            if cleaned and cleaned != vq:
                logger.info(f"Negation validation: cleaned vector_query "
                            f"'{vq}' → '{cleaned}'")
                result["vector_query"] = cleaned

            # Update negative_query if we extracted terms from VQ
            if neg_q and not result.get("negative_query", "").strip():
                result["negative_query"] = neg_q
                logger.info(f"Negation validation: built negative_query from VQ: '{neg_q}'")

        # Step 3: Remove negative nouns from fts_keywords
        neg_q = result.get("negative_query", "")
        excl = set(k.lower() for k in result.get("exclude_keywords", []))
        if excl and result.get("fts_keywords"):
            neg_extra = set(t.lower() for t in neg_q.split()) if neg_q else set()
            all_neg = excl | neg_extra
            cleaned_fts = [
                kw for kw in result["fts_keywords"]
                if not any(n in kw.lower() for n in all_neg)
            ]
            if cleaned_fts:
                result["fts_keywords"] = cleaned_fts

        return result

    def _augment_path_query(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Force path information to be used as searchable data.

        Path-like query signals:
        - contains '/' or '\\'
        - contains file extension (.psd/.png/.jpg/.jpeg)
        """
        q = (original_query or "").strip()
        if not q:
            return result

        has_sep = ('/' in q) or ('\\' in q)
        ext_match = re.search(r'\.(psd|png|jpg|jpeg)\b', q, flags=re.IGNORECASE)
        if not has_sep and not ext_match:
            return result

        norm = q.replace('\\', '/').strip().strip('"\'`')
        segments = [s for s in norm.split('/') if s]

        def _is_meaningful_path_token(tok: str) -> bool:
            t = str(tok).strip().lower()
            if len(t) < 2:
                return False
            if t.isdigit():
                return False
            if re.fullmatch(r'v\d+', t):
                return False
            if t in {"psd", "png", "jpg", "jpeg"}:
                return False
            if t in {"assets", "asset", "images", "image", "img", "files", "file", "data", "resource", "resources", "output", "outputs", "tmp", "temp", "test"}:
                return False
            return True

        # Build path keywords (segments + split tokens)
        path_keywords: List[str] = []
        for seg in segments:
            # Keep simple folder-like segments as whole words only.
            if re.fullmatch(r'[0-9a-zA-Z가-힣]{2,}', seg) and _is_meaningful_path_token(seg):
                path_keywords.append(seg)
            for part in re.split(r'[^0-9a-zA-Z가-힣]+', seg):
                part = part.strip()
                if _is_meaningful_path_token(part):
                    path_keywords.append(part)

        # Merge into fts_keywords (order-preserving dedupe)
        existing = result.get("fts_keywords", [])
        if not isinstance(existing, list):
            existing = [str(existing)]
        merged = []
        seen = set()
        for kw in existing + path_keywords:
            k = str(kw).strip()
            if k and k not in seen:
                seen.add(k)
                merged.append(k)
        result["fts_keywords"] = merged if merged else [q]

        # Add folder/file format filters when obvious
        filters = result.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}

        if segments:
            folder_hint = '/'.join(segments[:-1]) if ext_match and len(segments) > 1 else norm
            folder_hint = folder_hint.strip('/')
            if folder_hint and not filters.get("folder_path"):
                filters["folder_path"] = folder_hint

        if ext_match and not filters.get("format"):
            ext = ext_match.group(1).upper()
            filters["format"] = "JPG" if ext == "JPEG" else ext

        result["filters"] = filters

        # Path query is usually keyword-like
        result["query_type"] = "keyword"
        return result

    def _augment_ko_keywords(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Always augment FTS keywords with Korean keyword extraction.

        LLM (especially small 0.6B models) often fails to decompose Korean
        queries properly, passing them through as-is. This safety net ensures
        meaningful Korean keywords are extracted regardless of LLM quality.

        Does NOT override query_type when LLM already decomposed successfully —
        LLM may classify compound queries (e.g. folder filter + content search)
        as "visual" or "semantic", which is correct for mixed intent.
        """
        ko_kw = self._extract_ko_keywords(original_query)
        if not ko_kw:
            return result

        existing = set(result.get("fts_keywords", []))
        for kw in ko_kw:
            if kw not in existing:
                result["fts_keywords"].append(kw)

        # Only force keyword type when LLM didn't decompose (fallback path)
        if not result.get("decomposed", False):
            has_hangul = any('\uac00' <= c <= '\ud7af' for c in original_query)
            if has_hangul:
                result["query_type"] = "keyword"

        return result

    def _extract_ko_negation(self, query: str) -> List[str]:
        """Extract negated nouns from Korean text using regex patterns."""
        nouns = []
        for pattern in self._KO_NEG_PATTERNS:
            for m in pattern.finditer(query):
                noun = m.group(1).strip()
                if noun and len(noun) >= 1:
                    nouns.append(noun)
        return nouns

    @staticmethod
    def _translate_terms(terms: List[str]) -> List[str]:
        """Best-effort translate Korean terms to English."""
        try:
            from deep_translator import GoogleTranslator
            joined = ", ".join(terms)
            translated = GoogleTranslator(source='ko', target='en').translate(joined)
            if translated:
                return [t.strip().lower() for t in translated.split(",") if t.strip()]
        except Exception as e:
            logger.debug(f"Term translation failed: {e}")
        return []

    # Korean particles/suffixes to strip for FTS keyword extraction
    _KO_PARTICLES = re.compile(
        r'(자료중에|자료|폴더에|폴더|파일|이미지|중에서|중에|에서|에게|으로|이랑|에게서|부터|까지|한테|보다|처럼|만큼|대로|마다'
        r'|에|의|를|을|이|가|은|는|과|와|도|로|서|께|랑'
        r'|있는거|있는것|있는|없는|하는|했던|인거|인것|그려진거|보이는|보이는거)$'
    )
    _KO_STOPWORDS = {"있다", "있고", "있음", "찾아줘", "찾기"}

    @staticmethod
    def _is_meaningful_ko_token(token: str) -> bool:
        """Korean one-syllable nouns like 달/밤/성 are valid FTS literals."""
        return bool(token) and any('\uac00' <= ch <= '\ud7af' for ch in token)

    def _extract_ko_keywords(self, query: str) -> list:
        """Extract meaningful keywords from Korean text by stripping particles."""
        # Split on spaces and common delimiters
        tokens = re.split(r'[\s,./\\·]+', query.strip())
        keywords = []
        for tok in tokens:
            if not tok:
                continue
            # Strip trailing Korean particles
            cleaned = self._KO_PARTICLES.sub('', tok)
            if tok in self._KO_STOPWORDS or cleaned in self._KO_STOPWORDS:
                continue
            if cleaned and (len(cleaned) >= 2 or self._is_meaningful_ko_token(cleaned)):
                keywords.append(cleaned)
            elif tok and (len(tok) >= 2 or self._is_meaningful_ko_token(tok)):
                keywords.append(tok)
        return keywords

    def _fallback(self, query: str) -> Dict[str, Any]:
        """Fallback when LLM is unavailable - translate query for cross-language search."""
        vector_query = query
        # Extract individual keywords (not the whole query as one string)
        ko_keywords = self._extract_ko_keywords(query)
        fts_keywords = ko_keywords if ko_keywords else [query]

        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='auto', target='en').translate(query)
            if translated and translated.lower() != query.lower():
                vector_query = translated
                fts_keywords.append(translated)
                logger.info(f"Fallback translated: '{query}' → '{translated}'")
        except Exception as e:
            logger.debug(f"Fallback translation skipped: {e}")

        # Classify query type: if any Korean keyword looks like a proper noun
        # (project/folder name), treat as keyword search.
        query_type = "balanced"
        stripped = query.strip()
        has_cjk = any('\u3000' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in stripped)
        if has_cjk:
            query_type = "keyword"

        return {
            "vector_query": vector_query,
            "negative_query": "",
            "fts_keywords": fts_keywords,
            "exclude_keywords": [],
            "filters": {},
            "folder_filter": "",
            "query_type": query_type,
            "decomposed": False,
        }
