"""
Ingest Engine - Main Entry Point for Data Pipeline.

Handles:
1. File detection (CLI or Watchdog)
2. Parser selection (Factory Pattern)
3. Execution and Result Logging

v3.3: Phase-based batch pipeline with JIT thumbnail loading.
      Phase 1: Parse ALL (CPU parallel, no thumbnails) → Store metadata
      → Phase 2: VLM (sub-batch JIT load) → Store vision fields
      → Phase 3: Embed (sub-batch JIT load for VV, text-only MV) → Store vectors
"""

import argparse
import logging
import sys
import time
import io
import json as _json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
# Translation removed from ingest pipeline.
# Cross-language search is handled at query time by QueryDecomposer.

# Force UTF-8 for stdout/stderr to handle generic unicode characters
# line_buffering=True ensures each log line is flushed immediately
# (critical for subprocess → Electron IPC stdout parsing)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# Set process title for Activity Monitor visibility
try:
    import setproctitle
    setproctitle.setproctitle("Imagine-Pipeline")
except ImportError:
    pass

# Add project root to sys.path for module resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import Parsers
from backend.parser.base_parser import BaseParser
from backend.parser.psd_parser import PSDParser
from backend.parser.image_parser import ImageParser
from backend.parser.schema import AssetMeta
# Legacy: AutoBatchCalibrator kept for reference but no longer used in main pipeline
# from backend.utils.auto_batch_calibrator import AutoBatchCalibrator


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("IngestEngine")

SUPPORTED_EXTENSIONS = {'.psd', '.png', '.jpg', '.jpeg'}


@dataclass
class ParsedFile:
    """Intermediate state for Phase-based batch pipeline."""
    file_path: Path
    folder_path: str = None
    folder_depth: int = 0
    folder_tags: list = field(default_factory=list)
    meta: "AssetMeta" = None
    parser: "BaseParser" = None
    thumb_path: Path = None
    thumb_image_rgb: "Image.Image" = None  # Pre-loaded for Phase 2/3
    mc_raw: dict = None
    error: str = None
    # Per-phase smart skip flags (set after Phase 1 by DB check)
    skip_vision: bool = False
    skip_embed_vv: bool = False
    skip_embed_mv: bool = False
    # DB file ID (set by phase_store_metadata, used by phase_store_vectors)
    db_file_id: int = None


class ParserFactory:
    """Factory to select the correct parser for a file."""
    
    _parsers = [PSDParser, ImageParser]
    
    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[BaseParser]:
        """Return an instantiated parser capable of handling the file."""
        for parser_cls in cls._parsers:
            if parser_cls.can_parse(file_path):
                return parser_cls(output_dir=PROJECT_ROOT / "output")
        return None


class IngestHandler(FileSystemEventHandler):
    """Watchdog Handler for file events."""

    def __init__(self, watch_root: Path = None):
        super().__init__()
        self.watch_root = watch_root.resolve() if watch_root else None

    def _compute_folder_info(self, file_path: Path) -> Tuple[Optional[str], int, Optional[List[str]]]:
        """Compute folder metadata relative to watch root."""
        if self.watch_root:
            try:
                rel = file_path.resolve().parent.relative_to(self.watch_root)
                folder_str = str(rel).replace('\\', '/') if str(rel) != '.' else ''
                tags = [p for p in folder_str.split('/') if p] if folder_str else []
                depth = len(tags)
                return folder_str, depth, tags
            except ValueError:
                pass
        return None, 0, None

    def on_created(self, event):
        if not event.is_directory:
            fp = Path(event.src_path)
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                folder_path, depth, tags = self._compute_folder_info(fp)
                process_file(fp, folder_path=folder_path, folder_depth=depth, folder_tags=tags)

    def on_modified(self, event):
        if not event.is_directory:
            fp = Path(event.src_path)
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                folder_path, depth, tags = self._compute_folder_info(fp)
                process_file(fp, folder_path=folder_path, folder_depth=depth, folder_tags=tags)


def process_file(
    file_path: Path,
    folder_path: str = None,
    folder_depth: int = 0,
    folder_tags: list = None
):
    """Main processing logic for a single file."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Processing: {file_path}")

    # === STEP 1/4: Parsing ===
    step_start = time.time()
    logger.info(f"STEP 1/4 Parsing")

    # 1. Select Parser
    parser = ParserFactory.get_parser(file_path)
    if not parser:
        logger.warning(f"No parser found for: {file_path}")
        return

    # 2. Parse
    parse_start = time.time()
    result = parser.parse(file_path)
    parse_duration = time.time() - parse_start
    logger.info(f"  → Parsing completed in {parse_duration:.2f}s")

    # 3. Handle Result
    if result.success:
        meta = result.asset_meta

        # Inject folder discovery metadata
        if folder_path is not None:
            # --discover/--watch mode
            meta.folder_path = folder_path
            meta.folder_depth = folder_depth
            meta.folder_tags = folder_tags or []
        else:
            # --file/--files mode: extract folder metadata from file path
            # Use folder name only (not absolute path) for consistency with --discover mode
            parent = file_path.parent
            if parent.name and parent.name not in (".", ""):
                meta.folder_path = parent.name
                meta.folder_depth = 0
                meta.folder_tags = [parent.name]

        # === Extract Thumbnail Path (used by Vision & Vector Indexing) ===
        thumb_path = None
        if meta.thumbnail_url:
            if meta.thumbnail_url.startswith('file:///'):
                # URI format
                import urllib.parse
                p = urllib.parse.unquote(meta.thumbnail_url[8:])
                thumb_path = Path(p)
            else:
                # Relative or absolute path
                thumb_path = Path(meta.thumbnail_url)
                if not thumb_path.is_absolute():
                    # Make absolute relative to project root
                    thumb_path = Path(__file__).parent.parent.parent / thumb_path

        # === STEP 2/4: AI Vision (2-Stage) ===
        step2_start = time.time()
        logger.info(f"STEP 2/4 AI Vision (2-Stage)")

        # v3.1: Tier metadata (always set, regardless of Vision success/failure)
        from backend.utils.config import get_config as _get_config_tier
        from backend.utils.tier_config import get_active_tier as _get_active_tier

        _cfg_tier = _get_config_tier()
        _tier_name, _tier_config = _get_active_tier()

        meta.mode_tier = _tier_name
        meta.caption_model = _tier_config.get("vlm", {}).get("model", "")
        meta.text_embed_model = _tier_config.get("text_embed", {}).get("model", "")
        meta.runtime_version = _cfg_tier.get("runtime.ollama_version", "")
        meta.preprocess_params = _tier_config.get("preprocess", {})
        meta.embedding_model = _tier_config.get("visual", {}).get("model") or _cfg_tier.get(
            "embedding.visual.model", "google/siglip2-so400m-patch14-384"
        )
        meta.embedding_version = 1

        # === 4. AI Vision Analysis (v3 P0: 2-Stage Pipeline) ===
        try:
            if thumb_path and thumb_path.exists():
                logger.info("Running 2-Stage AI vision analysis...")
                vision_start = time.time()

                # Lazy load Vision Analyzer (environment-based factory)
                global _global_vision_analyzer
                if '_global_vision_analyzer' not in globals() or _global_vision_analyzer is None:
                    from backend.vision.vision_factory import get_vision_analyzer
                    _global_vision_analyzer = get_vision_analyzer()

                from PIL import Image
                import json as _json

                # v3.1: Load thumbnail (keep RGBA if present)
                thumb_img = Image.open(thumb_path)

                # Build MC.raw context before vision analysis
                mc_raw = {
                    "file_name": meta.file_name,
                    "folder_path": meta.folder_path or "",
                    "layer_names": meta.semantic_tags[:200] if meta.semantic_tags else [],  # First 200 chars
                    "used_fonts": meta.used_fonts[:5] if meta.used_fonts else [],  # First 5 fonts
                    "ocr_text": "",  # Will be filled by vision
                    "text_content": meta.text_content[:3] if meta.text_content else [],  # First 3 text layers
                }

                # Composite to RGB for vision model (in-memory, not saved)
                if thumb_img.mode == 'RGBA':
                    from backend.utils.config import get_config
                    cfg = get_config()
                    bg_color = cfg.get('thumbnail.index_composite_bg', '#FFFFFF')
                    # Convert hex to RGB
                    bg_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    background = Image.new('RGB', thumb_img.size, bg_rgb)
                    background.paste(thumb_img, mask=thumb_img.split()[-1])
                    image = background
                else:
                    image = thumb_img.convert("RGB")

                # v3.1: Compute perceptual hash (dHash) BEFORE vision analysis
                # (independent of vision success/failure)
                try:
                    from backend.utils.dhash import dhash64
                    hash_val = dhash64(image)
                    # Convert unsigned to signed (SQLite INTEGER is signed 64-bit)
                    if hash_val >= 2**63:
                        hash_val -= 2**64
                    meta.perceptual_hash = hash_val
                    logger.debug(f"   Perceptual hash: {meta.perceptual_hash}")
                except Exception as e:
                    logger.warning(f"   dHash calculation failed: {e}")
                    # Continue processing even if hash calculation fails

                # Check if adapter supports 2-stage
                if hasattr(_global_vision_analyzer, 'classify_and_analyze'):
                    # v3.1: 2-Stage with MC.raw context injection
                    vision_result = _global_vision_analyzer.classify_and_analyze(image, context=mc_raw)

                    # Common fields
                    meta.mc_caption = vision_result.get('caption', '')
                    meta.ai_tags = vision_result.get('tags', [])
                    meta.ocr_text = vision_result.get('ocr', '') or vision_result.get('text_content', '')
                    meta.dominant_color = vision_result.get('color', '') or vision_result.get('color_palette', '')
                    meta.ai_style = vision_result.get('style', '') or vision_result.get('art_style', '')

                    # v3 P0: structured fields
                    meta.image_type = vision_result.get('image_type')
                    meta.art_style = vision_result.get('art_style')
                    meta.color_palette = vision_result.get('color_palette')
                    meta.scene_type = vision_result.get('scene_type')
                    meta.time_of_day = vision_result.get('time_of_day')
                    meta.weather = vision_result.get('weather')
                    meta.character_type = vision_result.get('character_type')
                    meta.item_type = vision_result.get('item_type')
                    meta.ui_type = vision_result.get('ui_type')
                    meta.structured_meta = _json.dumps(vision_result, ensure_ascii=False)

                    logger.info(f"   Type: {meta.image_type}")
                else:
                    # Legacy single-pass fallback
                    vision_result = _global_vision_analyzer.analyze(image)
                    meta.mc_caption = vision_result.get('caption', '')
                    meta.ai_tags = vision_result.get('tags', [])
                    meta.ocr_text = vision_result.get('ocr', '')
                    meta.dominant_color = vision_result.get('color', '')
                    meta.ai_style = vision_result.get('style', '')

                vision_duration = time.time() - vision_start
                logger.info(f"  → AI Vision completed in {vision_duration:.2f}s")
                logger.info(f"   MC Caption: {meta.mc_caption[:80]}..." if len(meta.mc_caption or '') > 80 else f"   MC Caption: {meta.mc_caption}")
                logger.info(f"   AI Tags: {', '.join(meta.ai_tags[:10])}")
                if meta.ocr_text:
                    logger.info(f"   OCR: {meta.ocr_text[:50]}...")
                logger.info(f"   Color: {meta.dominant_color}")
            else:
                logger.debug("No thumbnail available for AI analysis")

        except Exception as e:
            logger.warning(f"AI Vision Analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Continue processing even if Vision fails

        step2_duration = time.time() - step2_start
        logger.info(f"STEP 2/4 completed in {step2_duration:.2f}s")

        # === Path normalization (always runs, regardless of Vision success) ===
        normalized = str(file_path).replace('\\', '/')
        for marker in ['/assets/', '/Assets/', '/resources/', '/Resources/']:
            idx = normalized.find(marker)
            if idx != -1:
                meta.storage_root = normalized[:idx]
                meta.relative_path = normalized[idx:].lstrip('/')
                break
        else:
            parts = normalized.rsplit('/', 1)
            meta.storage_root = parts[0] if len(parts) > 1 else ''
            meta.relative_path = parts[-1]

        # Save metadata JSON (always runs, regardless of Vision success)
        parser._save_json(meta, file_path)

        # === STEP 3/4: Embedding ===
        step3_start = time.time()
        logger.info(f"STEP 3/4 Embedding")
        # === 5. SQLite Storage (Vector + Metadata) ===
        try:
            # Lazy load SQLite client and SigLIP2 encoder
            global _global_sqlite_db, _global_clip_model

            if '_global_sqlite_db' not in globals() or _global_sqlite_db is None:
                from backend.db.sqlite_client import SQLiteDB
                _global_sqlite_db = SQLiteDB()
                logger.info("SQLite database initialized")

            if '_global_encoder' not in globals() or _global_encoder is None:
                from backend.vector.siglip2_encoder import SigLIP2Encoder
                _global_encoder = SigLIP2Encoder()
                logger.info(f"Embedding encoder loaded: {_global_encoder.model_name}")

            # Generate embedding from thumbnail
            embedding = None
            if thumb_path and thumb_path.exists():
                from PIL import Image
                import numpy as np

                try:
                    image = Image.open(thumb_path).convert("RGB")
                    embedding = _global_encoder.encode_image(image)
                    logger.debug(f"Generated embedding: {embedding.shape}")
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
                    embedding = np.zeros(_global_encoder.dimensions, dtype=np.float32)
            else:
                logger.warning("No thumbnail available - using zero embedding")
                import numpy as np
                from backend.utils.tier_config import get_active_tier as _get_tier
                _, _tc = _get_tier()
                dims = _tc.get("visual", {}).get("dimensions", 768)
                embedding = np.zeros(dims, dtype=np.float32)

            step3_duration = time.time() - step3_start
            logger.info(f"STEP 3/4 completed in {step3_duration:.2f}s")

            # === STEP 4/4: Storing ===
            step4_start = time.time()
            logger.info(f"STEP 4/4 Storing")
            logger.info(f"Storing to SQLite: {file_path.name}")
            file_id = _global_sqlite_db.insert_file(
                file_path=str(file_path),
                metadata=meta.model_dump(),
                embedding=embedding
            )

            # MV: Generate meaning vector from MC (caption + tags)
            try:
                from backend.utils.config import get_config as _get_cfg
                if _get_cfg().get("embedding.text.enabled"):
                    global _global_text_provider
                    if '_global_text_provider' not in globals() or _global_text_provider is None:
                        from backend.vector.text_embedding import get_text_embedding_provider
                        _global_text_provider = get_text_embedding_provider()

                    from backend.vector.text_embedding import build_document_text

                    # v3.1: Build MV document with [SEMANTIC]/[FACTS] format
                    facts = {
                        "image_type": meta.image_type,
                        "scene_type": meta.scene_type,
                        "art_style": meta.art_style,
                        "fonts": ", ".join(meta.used_fonts[:5]) if meta.used_fonts else None,
                        "path": meta.relative_path or meta.folder_path,
                    }
                    doc_text = build_document_text(meta.mc_caption, meta.ai_tags, facts=facts)
                    if doc_text:
                        text_emb = _global_text_provider.encode(doc_text)
                        if np.any(text_emb):
                            text_emb_list = text_emb.astype(np.float32).tolist()
                            cursor = _global_sqlite_db.conn.cursor()
                            cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (file_id,))
                            import json as _json2
                            cursor.execute(
                                "INSERT INTO vec_text (file_id, embedding) VALUES (?, ?)",
                                (file_id, _json2.dumps(text_emb_list))
                            )
                            _global_sqlite_db.conn.commit()
                            logger.info(f"   MV stored ({len(text_emb)}-dim)")
            except Exception as e:
                logger.warning(f"MV generation failed (non-fatal): {e}")

        except Exception as e:
            logger.error(f"SQLite Storage Failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        step4_duration = time.time() - step4_start
        logger.info(f"STEP 4/4 completed in {step4_duration:.2f}s")

        total_duration = time.time() - step_start
        logger.info(f"[OK] Parsed successfully in {total_duration:.2f}s")
        logger.info(f"   Format: {meta.format}, Layers: {meta.layer_count}")
        logger.info(f"   Tags: {meta.semantic_tags}")
        if meta.thumbnail_url:
            logger.info(f"   Thumb: {meta.thumbnail_url}")
    else:
        logger.error(f"[FAIL] Parsing failed: {result.errors}")


# ── Phase-based Batch Pipeline (v3.2) ───────────────────────────────────

def _resolve_thumb_path(meta) -> Optional[Path]:
    """Extract thumbnail path from AssetMeta."""
    if not meta.thumbnail_url:
        return None
    if meta.thumbnail_url.startswith('file:///'):
        import urllib.parse
        p = urllib.parse.unquote(meta.thumbnail_url[8:])
        return Path(p)
    thumb_path = Path(meta.thumbnail_url)
    if not thumb_path.is_absolute():
        thumb_path = PROJECT_ROOT / thumb_path
    return thumb_path


def _load_and_composite_thumbnail(thumb_path: Path) -> Optional["Image.Image"]:
    """Load thumbnail and composite RGBA to RGB."""
    from PIL import Image
    thumb_img = Image.open(thumb_path)
    if thumb_img.mode == 'RGBA':
        from backend.utils.config import get_config
        cfg = get_config()
        bg_color = cfg.get('thumbnail.index_composite_bg', '#FFFFFF')
        bg_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        background = Image.new('RGB', thumb_img.size, bg_rgb)
        background.paste(thumb_img, mask=thumb_img.split()[-1])
        return background
    return thumb_img.convert("RGB")


def _build_mc_raw(meta) -> dict:
    """Build MC.raw context dict for VLM Stage 2."""
    return {
        "file_name": meta.file_name,
        "folder_path": meta.folder_path or "",
        "layer_names": meta.semantic_tags[:200] if meta.semantic_tags else [],
        "used_fonts": meta.used_fonts[:5] if meta.used_fonts else [],
        "ocr_text": "",
        "text_content": meta.text_content[:3] if meta.text_content else [],
    }


def _set_tier_metadata(meta):
    """Set tier-related metadata fields on AssetMeta."""
    from backend.utils.config import get_config
    from backend.utils.tier_config import get_active_tier

    cfg = get_config()
    tier_name, tier_config = get_active_tier()

    meta.mode_tier = tier_name
    meta.caption_model = tier_config.get("vlm", {}).get("model", "")
    meta.text_embed_model = tier_config.get("text_embed", {}).get("model", "")
    meta.runtime_version = cfg.get("runtime.ollama_version", "")
    meta.preprocess_params = tier_config.get("preprocess", {})
    meta.embedding_model = tier_config.get("visual", {}).get("model") or cfg.get(
        "embedding.visual.model", "google/siglip2-so400m-patch14-384"
    )
    meta.embedding_version = 1


def _normalize_paths(meta, file_path: Path):
    """Normalize storage_root and relative_path."""
    normalized = str(file_path).replace('\\', '/')
    for marker in ['/assets/', '/Assets/', '/resources/', '/Resources/']:
        idx = normalized.find(marker)
        if idx != -1:
            meta.storage_root = normalized[:idx]
            meta.relative_path = normalized[idx:].lstrip('/')
            return
    parts = normalized.rsplit('/', 1)
    meta.storage_root = parts[0] if len(parts) > 1 else ''
    meta.relative_path = parts[-1]


def _apply_vision_result(meta, vision_result: dict):
    """Apply VLM result fields to AssetMeta."""
    meta.mc_caption = vision_result.get('caption', '')
    meta.ai_tags = vision_result.get('tags', [])
    meta.ocr_text = vision_result.get('ocr', '') or vision_result.get('text_content', '')
    meta.dominant_color = vision_result.get('color', '') or vision_result.get('color_palette', '')
    meta.ai_style = vision_result.get('style', '') or vision_result.get('art_style', '')
    # Structured fields
    meta.image_type = vision_result.get('image_type')
    meta.art_style = vision_result.get('art_style')
    meta.color_palette = vision_result.get('color_palette')
    meta.scene_type = vision_result.get('scene_type')
    meta.time_of_day = vision_result.get('time_of_day')
    meta.weather = vision_result.get('weather')
    meta.character_type = vision_result.get('character_type')
    meta.item_type = vision_result.get('item_type')
    meta.ui_type = vision_result.get('ui_type')
    meta.structured_meta = _json.dumps(vision_result, ensure_ascii=False)


def _parse_single_file(file_info: tuple) -> ParsedFile:
    """Parse a single file (CPU-bound). Used by Phase 1 ThreadPool."""
    fp, folder, depth, tags = file_info

    pf = ParsedFile(file_path=fp, folder_path=folder, folder_depth=depth, folder_tags=tags or [])

    if not fp.exists():
        pf.error = f"File not found: {fp}"
        return pf

    parser = ParserFactory.get_parser(fp)
    if not parser:
        pf.error = f"No parser found for: {fp}"
        return pf

    try:
        result = parser.parse(fp)
    except Exception as e:
        pf.error = f"Parse error: {e}"
        return pf

    if not result.success:
        pf.error = f"Parsing failed: {result.errors}"
        return pf

    meta = result.asset_meta
    pf.meta = meta
    pf.parser = parser

    # Inject folder metadata
    if folder is not None:
        meta.folder_path = folder
        meta.folder_depth = depth
        meta.folder_tags = tags or []
    else:
        parent = fp.parent
        if parent.name and parent.name not in (".", ""):
            meta.folder_path = parent.name
            meta.folder_depth = 0
            meta.folder_tags = [parent.name]

    # Resolve thumbnail (path only, no image loading — JIT in Phase 2/3)
    pf.thumb_path = _resolve_thumb_path(meta)

    # Build MC.raw context
    pf.mc_raw = _build_mc_raw(meta)

    # Set tier metadata
    _set_tier_metadata(meta)

    # Normalize paths
    _normalize_paths(meta, fp)

    return pf


def phase1_parse_all(
    file_infos: List[tuple], max_workers: int = 4
) -> List[ParsedFile]:
    """Phase 1: CPU-parallel parsing, metadata extraction (no thumbnail loading)."""
    t0 = time.perf_counter()
    logger.info(f"STEP 1/4 Parsing ({len(file_infos)} files, workers={max_workers})")

    if max_workers > 1 and len(file_infos) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_parse_single_file, file_infos))
    else:
        results = [_parse_single_file(fi) for fi in file_infos]

    ok = sum(1 for r in results if r.error is None)
    fail = sum(1 for r in results if r.error is not None)
    elapsed = time.perf_counter() - t0
    logger.info(f"STEP 1/4 completed in {elapsed:.1f}s ({ok} ok, {fail} errors)")

    for pf in results:
        if pf.error:
            logger.warning(f"  [FAIL] {pf.file_path.name}: {pf.error}")

    return results


def _compute_dhash(pf: ParsedFile):
    """Compute perceptual dHash from loaded thumbnail."""
    if pf.thumb_image_rgb is None:
        return
    try:
        from backend.utils.dhash import dhash64
        hash_val = dhash64(pf.thumb_image_rgb)
        if hash_val >= 2**63:
            hash_val -= 2**64
        pf.meta.perceptual_hash = hash_val
    except Exception as e:
        logger.warning(f"dHash failed for {pf.file_path.name}: {e}")


def phase2_vision_all(
    parsed_files: List[ParsedFile], progress_callback=None,
    phase_progress_callback=None
) -> None:
    """Phase 2: VLM analysis with sub-batch JIT thumbnail loading."""
    from backend.utils.tier_config import get_active_tier
    _, tier_config = get_active_tier()
    sub_batch_size = tier_config.get("vlm", {}).get("batch_size", 8)

    # Filter files that need vision processing
    vision_indices = []
    skipped = 0
    for idx, pf in enumerate(parsed_files):
        if pf.error is not None:
            continue
        if pf.skip_vision:
            skipped += 1
            logger.info(f"  [SKIP:Vision] {pf.file_path.name} (MC exists, same model)")
            continue
        if pf.thumb_path and pf.thumb_path.exists():
            vision_indices.append(idx)

    if skipped > 0:
        logger.info(f"  [SKIP:Vision] {skipped} files skipped (MC unchanged)")

    if not vision_indices:
        logger.info("STEP 2/4 AI Vision — no images to analyze")
        return

    t0 = time.perf_counter()
    total_vision = len(vision_indices)
    logger.info(f"STEP 2/4 AI Vision ({total_vision} images)")

    from backend.vision.vision_factory import get_vision_analyzer
    analyzer = get_vision_analyzer()

    global_progress_idx = 0

    # Process in sub-batches: JIT load → dHash → VLM → release
    for chunk_start in range(0, total_vision, sub_batch_size):
        chunk = vision_indices[chunk_start:chunk_start + sub_batch_size]

        # JIT: Load thumbnails for this sub-batch only
        for idx in chunk:
            pf = parsed_files[idx]
            try:
                pf.thumb_image_rgb = _load_and_composite_thumbnail(pf.thumb_path)
            except Exception as e:
                logger.warning(f"Thumbnail load failed for {pf.file_path.name}: {e}")
            _compute_dhash(pf)

        # Build vision items for this sub-batch
        vision_items = []
        valid_chunk = []
        for idx in chunk:
            pf = parsed_files[idx]
            if pf.thumb_image_rgb is not None:
                vision_items.append((pf.thumb_image_rgb, pf.mc_raw))
                valid_chunk.append(idx)

        if not vision_items:
            continue

        # Progress callback wired to global index
        def _progress(i, total, result, _offset=global_progress_idx):
            pf = parsed_files[valid_chunk[i]]
            logger.info(f"  [{_offset + i + 1}/{total_vision}] {pf.file_path.name} → {result.get('image_type', '?')}")
            if progress_callback:
                progress_callback(_offset + i, total_vision, result)

        if hasattr(analyzer, 'classify_and_analyze_sequence'):
            vision_results = analyzer.classify_and_analyze_sequence(
                vision_items, progress_callback=_progress
            )
        else:
            vision_results = []
            for i, (img, ctx) in enumerate(vision_items):
                try:
                    r = analyzer.classify_and_analyze(img, context=ctx)
                except Exception:
                    r = {"caption": "", "tags": [], "image_type": "other"}
                vision_results.append(r)
                _progress(i, len(vision_items), r)

        # Apply results + release thumbnails
        for i, result in enumerate(vision_results):
            pf = parsed_files[valid_chunk[i]]
            _apply_vision_result(pf.meta, result)
            logger.info(
                f"   MC Caption: {(pf.meta.mc_caption or '')[:80]}"
                + ("..." if len(pf.meta.mc_caption or '') > 80 else "")
            )

        # Release thumbnails immediately after VLM inference
        for idx in chunk:
            parsed_files[idx].thumb_image_rgb = None

        global_progress_idx += len(valid_chunk)

        # Force GC to reclaim CPU memory from processor intermediates
        import gc; gc.collect()

        # Emit intermediate phase progress after each sub-batch
        if phase_progress_callback:
            phase_progress_callback(global_progress_idx)

    elapsed = time.perf_counter() - t0
    logger.info(f"STEP 2/4 completed in {elapsed:.1f}s")


def phase3_embed_all(
    parsed_files: List[ParsedFile], vv_batch: int = None, mv_batch: int = None,
    phase_progress_callback=None
) -> List[tuple]:
    """
    Phase 3: True batch embedding (VV + MV).

    vv_batch=None uses adaptive batch sizing (auto-discovery inside encoder).
    vv_batch=int forces a specific batch size.

    Returns list of (vv_vec, mv_vec, mv_text) tuples aligned with parsed_files.
    Failed entries get zero vectors.
    """
    import numpy as np

    t0 = time.perf_counter()

    # Collect valid files for embedding
    valid_indices = [i for i, pf in enumerate(parsed_files) if pf.error is None]
    total = len(parsed_files)

    logger.info(f"STEP 3/4 Embedding ({len(valid_indices)} files)")

    # Initialize result list with zero placeholders
    from backend.utils.tier_config import get_active_tier
    _, tier_config = get_active_tier()
    vv_dims = tier_config.get("visual", {}).get("dimensions", 768)

    from backend.utils.config import get_config
    cfg = get_config()
    text_enabled = cfg.get("embedding.text.enabled", False)

    # Default zeros
    results = [(
        np.zeros(vv_dims, dtype=np.float32),
        None,
        None
    ) for _ in range(total)]

    # === VV: SigLIP2 batch encoding with JIT thumbnail loading ===
    vv_indices = []
    vv_skipped = 0
    for i in valid_indices:
        pf = parsed_files[i]
        if pf.skip_embed_vv:
            vv_skipped += 1
            continue
        if pf.thumb_path and pf.thumb_path.exists():
            vv_indices.append(i)
    if vv_skipped > 0:
        logger.info(f"  [SKIP:VV] {vv_skipped} files skipped (same SigLIP2 model)")

    if vv_indices:
        from backend.vector.siglip2_encoder import SigLIP2Encoder
        global _global_encoder
        if '_global_encoder' not in globals() or _global_encoder is None:
            _global_encoder = SigLIP2Encoder()
            logger.info(f"SigLIP2 encoder loaded: {_global_encoder.model_name}")

        # Determine sub-batch size for JIT loading
        vv_sub_batch = vv_batch or 8
        vv_encoded = 0

        for chunk_start in range(0, len(vv_indices), vv_sub_batch):
            chunk = vv_indices[chunk_start:chunk_start + vv_sub_batch]

            # JIT: Load thumbnails from disk for this sub-batch
            images = []
            chunk_valid = []
            for idx in chunk:
                pf = parsed_files[idx]
                try:
                    img = _load_and_composite_thumbnail(pf.thumb_path)
                    images.append(img)
                    chunk_valid.append(idx)
                except Exception as e:
                    logger.warning(f"VV thumbnail load failed for {pf.file_path.name}: {e}")

            if images:
                vv_vectors = _global_encoder.encode_image_batch(images)
                for j, vec in enumerate(vv_vectors):
                    idx = chunk_valid[j]
                    vv, mv, mv_text = results[idx]
                    results[idx] = (vec, mv, mv_text)
                vv_encoded += len(images)
            # images go out of scope → GC eligible

            # Emit intermediate phase progress after each VV sub-batch
            if phase_progress_callback:
                phase_progress_callback(vv_encoded, 'vv')

        logger.info(f"  VV: {vv_encoded} images encoded")

        # Unload SigLIP2 after VV encoding to free GPU memory before MV
        _global_encoder.unload()
        _global_encoder = None

    # === MV: Text embedding batch encoding ===
    if text_enabled:
        from backend.vector.text_embedding import get_text_embedding_provider, build_document_text

        global _global_text_provider
        if '_global_text_provider' not in globals() or _global_text_provider is None:
            _global_text_provider = get_text_embedding_provider()

        mv_texts = []
        mv_indices = []
        mv_skipped = 0
        for i in valid_indices:
            pf = parsed_files[i]
            if pf.skip_embed_mv:
                mv_skipped += 1
                continue
            meta = pf.meta
            facts = {
                "image_type": meta.image_type,
                "scene_type": meta.scene_type,
                "art_style": meta.art_style,
                "fonts": ", ".join(meta.used_fonts[:5]) if meta.used_fonts else None,
                "path": meta.relative_path or meta.folder_path,
            }
            doc_text = build_document_text(meta.mc_caption, meta.ai_tags, facts=facts)
            if doc_text:
                mv_texts.append(doc_text)
                mv_indices.append(i)

        if mv_skipped > 0:
            logger.info(f"  [SKIP:MV] {mv_skipped} files skipped (same text embed model)")

        if mv_texts:
            if mv_batch is None:
                mv_batch = 16  # Text embedding is lightweight, safe default

            try:
                mv_vectors = []
                if hasattr(_global_text_provider, 'encode_batch'):
                    # Sub-batch MV encoding
                    for sb_start in range(0, len(mv_texts), mv_batch):
                        sub = mv_texts[sb_start:sb_start + mv_batch]
                        mv_vectors.extend(_global_text_provider.encode_batch(sub))
                        if phase_progress_callback:
                            phase_progress_callback(len(mv_vectors), 'mv')
                else:
                    mv_vectors = [_global_text_provider.encode(t) for t in mv_texts]

                for j, vec in enumerate(mv_vectors):
                    idx = mv_indices[j]
                    vv, _, _ = results[idx]
                    results[idx] = (vv, vec, mv_texts[j])

                logger.info(f"  MV batch: {len(mv_vectors)} texts encoded (batch_size={mv_batch})")
            except Exception as e:
                logger.warning(f"  MV batch encoding failed: {e}")
                # Fallback: individual encoding
                for j, text in enumerate(mv_texts):
                    try:
                        vec = _global_text_provider.encode(text)
                        idx = mv_indices[j]
                        vv, _, _ = results[idx]
                        results[idx] = (vv, vec, text)
                    except Exception:
                        pass

    elapsed = time.perf_counter() - t0
    logger.info(f"STEP 3/4 completed in {elapsed:.1f}s")

    return results


def phase_store_metadata(parsed_files: List[ParsedFile]) -> int:
    """Phase 1 storage: INSERT basic metadata + save JSON files."""
    from backend.db.sqlite_client import SQLiteDB
    global _global_sqlite_db
    if '_global_sqlite_db' not in globals() or _global_sqlite_db is None:
        _global_sqlite_db = SQLiteDB()
        logger.info("SQLite database initialized")

    stored = 0
    for pf in parsed_files:
        if pf.error is not None:
            continue
        try:
            pf.parser._save_json(pf.meta, pf.file_path)
            pf.db_file_id = _global_sqlite_db.upsert_metadata(
                file_path=str(pf.file_path),
                metadata=pf.meta.model_dump()
            )
            stored += 1
        except Exception as e:
            logger.error(f"  [FAIL:meta] {pf.file_path.name}: {e}")
            pf.error = f"Metadata store failed: {e}"

    logger.info(f"  Phase 1 stored: {stored} metadata records")
    return stored


def phase_store_vision(parsed_files: List[ParsedFile]) -> int:
    """Phase 2 storage: UPDATE VLM-generated fields only."""
    global _global_sqlite_db

    stored = 0
    for pf in parsed_files:
        if pf.error is not None or pf.skip_vision:
            continue
        try:
            fields = {
                'mc_caption': pf.meta.mc_caption,
                'ai_tags': pf.meta.ai_tags,
                'ocr_text': pf.meta.ocr_text,
                'dominant_color': pf.meta.dominant_color,
                'ai_style': pf.meta.ai_style,
                'image_type': pf.meta.image_type,
                'art_style': pf.meta.art_style,
                'color_palette': pf.meta.color_palette,
                'scene_type': pf.meta.scene_type,
                'time_of_day': pf.meta.time_of_day,
                'weather': pf.meta.weather,
                'character_type': pf.meta.character_type,
                'item_type': pf.meta.item_type,
                'ui_type': pf.meta.ui_type,
                'structured_meta': pf.meta.structured_meta,
                'perceptual_hash': pf.meta.perceptual_hash,
            }
            _global_sqlite_db.update_vision_fields(str(pf.file_path), fields)
            # Re-save JSON with updated VLM fields
            pf.parser._save_json(pf.meta, pf.file_path)
            stored += 1
        except Exception as e:
            logger.error(f"  [FAIL:vision] {pf.file_path.name}: {e}")

    logger.info(f"  Phase 2 stored: {stored} vision records")
    return stored


def phase_store_vectors(parsed_files: List[ParsedFile], embeddings: List[tuple]) -> Tuple[int, int]:
    """Phase 3 storage: INSERT VV + MV vectors."""
    import numpy as np
    global _global_sqlite_db

    stored, errors = 0, 0
    for i, pf in enumerate(parsed_files):
        if pf.error is not None:
            continue

        vv_vec, mv_vec, _mv_text = embeddings[i]
        file_id = pf.db_file_id
        if file_id is None:
            logger.warning(f"  [SKIP:vec] {pf.file_path.name}: no db_file_id")
            continue

        try:
            vv = vv_vec if (vv_vec is not None and np.any(vv_vec) and not pf.skip_embed_vv) else None
            mv = mv_vec if (mv_vec is not None and np.any(mv_vec) and not pf.skip_embed_mv) else None

            if vv is not None or mv is not None:
                _global_sqlite_db.upsert_vectors(file_id, vv_vec=vv, mv_vec=mv)

            stored += 1
            logger.info(f"  [OK] {pf.file_path.name}")
        except Exception as e:
            errors += 1
            logger.error(f"  [FAIL:vec] {pf.file_path.name}: {e}")

    logger.info(f"  Phase 3 stored: {stored} vector records, {errors} errors")
    return stored, errors


def _check_phase_skip(parsed_files: List[ParsedFile]):
    """
    Per-phase smart skip: check DB for already-completed phase outputs.

    Sets skip_vision, skip_embed_vv, skip_embed_mv flags on each ParsedFile.
    For vision-skipped files, injects stored MC data into meta for Phase 3 MV encoding.

    Skip conditions (ALL must match):
    - File exists in DB with same mtime (content unchanged)
    - Phase output exists (mc_caption, vec_files, vec_text)
    - Model used matches current tier's model
    """
    from backend.db.sqlite_client import SQLiteDB
    from backend.utils.tier_config import get_active_tier

    global _global_sqlite_db
    if '_global_sqlite_db' not in globals() or _global_sqlite_db is None:
        _global_sqlite_db = SQLiteDB()

    _, tier_config = get_active_tier()
    current_vlm = tier_config.get("vlm", {}).get("model", "")
    current_vv_model = tier_config.get("visual", {}).get("model", "")
    current_mv_model = tier_config.get("text_embed", {}).get("model", "")

    skip_v, skip_vv, skip_mv = 0, 0, 0

    for pf in parsed_files:
        if pf.error is not None:
            continue

        info = _global_sqlite_db.get_file_phase_info(str(pf.file_path.resolve()))
        if info is None:
            continue  # New file, no skip possible

        # mtime check: if file content changed, run ALL phases
        stored_mtime = info.get("modified_at")
        if stored_mtime:
            current_mtime = datetime.fromtimestamp(pf.file_path.stat().st_mtime).isoformat()
            if stored_mtime.replace('T', ' ') != current_mtime.replace('T', ' '):
                continue  # File changed → run all phases

        # Vision: skip if MC exists with same VLM model
        if info["has_mc"] and info.get("caption_model") == current_vlm:
            pf.skip_vision = True
            # Inject stored MC into meta so Phase 3 MV can use it
            pf.meta.mc_caption = info["mc_caption"]
            if info.get("ai_tags"):
                try:
                    tags = info["ai_tags"]
                    pf.meta.ai_tags = _json.loads(tags) if isinstance(tags, str) else tags
                except Exception:
                    pass
            pf.meta.image_type = info.get("image_type")
            pf.meta.scene_type = info.get("scene_type")
            pf.meta.art_style = info.get("art_style")
            skip_v += 1

        # VV: skip if vector exists with same SigLIP2 model
        if info["has_vv"] and info.get("embedding_model") == current_vv_model:
            pf.skip_embed_vv = True
            skip_vv += 1

        # MV: skip if vector exists with same text embed model
        if info["has_mv"] and info.get("text_embed_model") == current_mv_model:
            pf.skip_embed_mv = True
            skip_mv += 1

    if skip_v > 0 or skip_vv > 0 or skip_mv > 0:
        logger.info(
            f"[SKIP:Phase] Vision: {skip_v}, VV: {skip_vv}, MV: {skip_mv} "
            f"files can skip (model+mtime match)"
        )


def process_batch_phased(
    file_infos: List[tuple],
    vv_batch: int = None,
    mv_batch: int = None,
    parse_workers: int = 4,
    progress_callback=None,
) -> Tuple[int, int, int]:
    """
    Phase-based batch pipeline: full Phase separation with JIT thumbnail loading.

    Phase 1: Parse ALL files (CPU parallel, no thumbnail loading)
    → Store metadata to DB
    Phase 2: VLM analysis (sub-batch JIT thumbnail load → infer → release)
    → Store vision fields to DB
    Phase 3: Embedding (sub-batch JIT thumbnail load for VV, text-only for MV)
    → Store vectors to DB

    Thumbnails are loaded from SSD (~1ms) per sub-batch and immediately released,
    limiting peak memory to ~14MB (batch_size × 1.7MB) instead of N × 1.7MB.

    Args:
        file_infos: List of (file_path, folder, depth, tags) tuples
        vv_batch: SigLIP2 sub-batch size (None=auto-calibrate)
        mv_batch: Text embedding sub-batch size (None=default 16)
        parse_workers: Phase 1 CPU thread count
        progress_callback: Optional fn(phase, idx, total, detail)

    Returns:
        (stored, skipped_or_failed, errors) counts
    """
    t_total = time.perf_counter()
    total = len(file_infos)

    logger.info(f"[BATCH] Phase-based pipeline v3.3: {total} files (sequential model load + MPS GC)")

    # Cumulative phase counters (emitted as [PHASE] for frontend tracking)
    cum_parse = 0
    cum_vision = 0
    cum_embed = 0
    cum_store = 0
    active_batch_info = ""  # e.g. "B:8" for current sub-batch size

    def _emit_phase_progress():
        """Emit cumulative phase progress for frontend."""
        logger.info(
            f"[PHASE] P:{cum_parse} V:{cum_vision} E:{cum_embed} S:{cum_store} "
            f"T:{total} {active_batch_info}"
        )

    # Phase 1: Parse ALL (CPU parallel, metadata only — no thumbnails)
    parsed = phase1_parse_all(file_infos, max_workers=parse_workers)
    valid_count = sum(1 for pf in parsed if pf.error is None)
    cum_parse = valid_count
    _emit_phase_progress()

    # Store Phase 1: metadata + JSON
    phase_store_metadata(parsed)

    # Per-phase smart skip: check DB for already-completed phases
    _check_phase_skip(parsed)

    # Phase 2: VLM (sub-batch JIT thumbnail loading)
    from backend.utils.tier_config import get_active_tier
    _, _tier_cfg = get_active_tier()
    vlm_bs = _tier_cfg.get("vlm", {}).get("batch_size", 8)

    def _on_vision_progress(count):
        nonlocal cum_vision, active_batch_info
        cum_vision = count
        active_batch_info = f"B:{vlm_bs}"
        _emit_phase_progress()

    phase2_vision_all(parsed, progress_callback=progress_callback,
                      phase_progress_callback=_on_vision_progress)
    cum_vision = valid_count
    active_batch_info = ""
    _emit_phase_progress()

    # Store Phase 2: vision fields only
    phase_store_vision(parsed)

    # Unload VLM to free VRAM before loading embedding models
    try:
        from backend.vision.vision_factory import get_vision_analyzer, VisionAnalyzerFactory
        analyzer = get_vision_analyzer()
        if hasattr(analyzer, 'unload_model'):
            analyzer.unload_model()
        VisionAnalyzerFactory.reset()  # Clear singleton so it can be re-created if needed
        logger.info("VLM unloaded after Phase 2 to free memory for Phase 3")
    except Exception:
        pass

    # Phase 3: Embed (sub-batch JIT thumbnail loading for VV, text-only for MV)
    def _on_embed_progress(count, kind):
        nonlocal cum_embed, active_batch_info
        cum_embed = count
        bs = vv_batch or 8 if kind == 'vv' else mv_batch or 16
        active_batch_info = f"B:{bs}:{kind.upper()}"
        _emit_phase_progress()

    embeddings = phase3_embed_all(parsed, vv_batch=vv_batch, mv_batch=mv_batch,
                                  phase_progress_callback=_on_embed_progress)
    cum_embed = valid_count
    active_batch_info = ""
    _emit_phase_progress()

    # Store Phase 3: VV + MV vectors
    stored, store_errors = phase_store_vectors(parsed, embeddings)
    cum_store = stored
    _emit_phase_progress()

    parse_errors = sum(1 for pf in parsed if pf.error is not None)

    elapsed = time.perf_counter() - t_total
    logger.info(
        f"[DONE] {stored} processed, {parse_errors} parse-errors, "
        f"{store_errors} store-errors (total: {total}, {elapsed:.1f}s)"
    )

    return stored, parse_errors, store_errors


def discover_files(root_dir: Path) -> List[Tuple[Path, str, int, List[str]]]:
    """
    DFS recursive discovery of supported image files.

    Args:
        root_dir: Root directory to scan

    Returns:
        List of (file_path, relative_folder, depth, folder_tags) tuples
    """
    discovered = []
    root_dir = root_dir.resolve()

    _skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vs'}

    def _dfs(current_dir: Path, depth: int):
        try:
            entries = sorted(
                current_dir.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower())
            )
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot read directory: {current_dir}: {e}")
            return

        for entry in entries:
            if entry.is_dir():
                if entry.name.startswith('.') or entry.name in _skip_dirs:
                    continue
                _dfs(entry, depth + 1)
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                rel = entry.parent.relative_to(root_dir)
                folder_str = str(rel).replace('\\', '/') if str(rel) != '.' else ''
                folder_tags = [p for p in folder_str.split('/') if p] if folder_str else []
                discovered.append((entry, folder_str, depth, folder_tags))

    _dfs(root_dir, 0)
    return discovered


def should_skip_file(file_path: Path, db) -> bool:
    """
    Compare DB modified_at with file's current mtime.
    Returns True if unchanged (should skip).
    """
    stored = db.get_file_modified_at(str(file_path.resolve()))
    if stored is None:
        return False
    current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    # Normalize: DB may store "2024-01-01 12:00:00", isoformat gives "2024-01-01T12:00:00"
    stored_normalized = stored.replace('T', ' ')
    current_normalized = current_mtime.replace('T', ' ')
    return stored_normalized == current_normalized


def should_skip_file_enhanced(file_path: Path, db, current_tier: str) -> bool:
    """
    Enhanced skip: mtime comparison + tier change detection.
    Returns True if file is unchanged AND was processed with the same tier.
    """
    resolved = str(file_path.resolve())
    stored = db.get_file_modified_at(resolved)
    if stored is None:
        return False  # Not in DB → must process

    # mtime comparison
    current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    stored_normalized = stored.replace('T', ' ')
    current_normalized = current_mtime.replace('T', ' ')
    if stored_normalized != current_normalized:
        return False  # File changed → must reprocess

    # Tier comparison: if DB tier differs from current, reprocess
    stored_tier = db.get_file_mode_tier(resolved)
    if stored_tier and stored_tier != current_tier:
        return False  # Tier changed → must reprocess

    return True  # Unchanged + same tier → skip


def run_discovery(root_dir: str, skip_processed: bool = True, batch_size: int = 5):
    """
    DFS discover all supported files and process them.

    v3.2: Uses Phase-based batch pipeline for 2+ files.

    Args:
        root_dir: Root directory to scan
        skip_processed: Skip files unchanged since last processing
        batch_size: (Legacy, ignored) Batch size parameter kept for API compat
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Directory not found: {root_dir}")
        return

    # DFS discovery
    logger.info(f"[DISCOVER] Scanning: {root_path}")
    discovered = discover_files(root_path)
    total = len(discovered)
    logger.info(f"[DISCOVER] Found {total} supported files")

    if total == 0:
        return

    # Filter files (smart skip)
    db = None
    if skip_processed:
        try:
            from backend.db.sqlite_client import SQLiteDB
            db = SQLiteDB()
        except Exception as e:
            logger.warning(f"Cannot open DB for smart skip: {e}")

    to_process = []
    skipped = 0
    for fp, folder, depth, tags in discovered:
        if skip_processed and db and should_skip_file(fp, db):
            skipped += 1
            logger.debug(f"[SKIP] {fp.name} (unchanged)")
        else:
            to_process.append((fp, folder, depth, tags))

    if skipped > 0:
        logger.info(f"[SKIP] {skipped} files unchanged, {len(to_process)} to process")

    if not to_process:
        logger.info("[DONE] No files to process")
        return

    # v3.2: Phase-based batch pipeline
    if len(to_process) > 1:
        stored, parse_errors, store_errors = process_batch_phased(to_process)
        logger.info(f"[DONE] {stored} stored, {skipped} skipped, "
                     f"{parse_errors + store_errors} errors (total: {total})")
    else:
        # Single file: use original process_file for simplicity
        fp, folder, depth, tags = to_process[0]
        logger.info(f"Processing: {fp.name}")
        try:
            process_file(fp, folder_path=folder, folder_depth=depth, folder_tags=tags)
            logger.info(f"[DONE] 1 processed, {skipped} skipped (total: {total})")
        except Exception as e:
            logger.error(f"[ERROR] {fp.name}: {e}")
            logger.info(f"[DONE] 0 processed, {skipped} skipped, 1 error (total: {total})")


def start_watcher(path: str):
    """Start directory watcher with initial scan."""
    path_obj = Path(path)
    if not path_obj.exists():
        logger.error(f"Directory not found: {path}")
        return

    # Initial scan of existing files
    logger.info(f"[WATCH] Initial scan of existing files...")
    run_discovery(path, skip_processed=True)

    # Start real-time file watching
    event_handler = IngestHandler(watch_root=path_obj)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    logger.info(f"[WATCH] Now watching for changes: {path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def parse_batch_size(value):
    """Parse batch size argument (int, 'auto', or 'adaptive')."""
    value_lower = value.lower()
    if value_lower in ['auto', 'adaptive']:
        return value_lower
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"batch-size must be an integer, 'auto', or 'adaptive', got: {value}")


def main():
    parser = argparse.ArgumentParser(description="ImageParser Ingest Engine")
    parser.add_argument("--file", help="Process a single file")
    parser.add_argument("--files", help="Process multiple files (JSON array)")
    parser.add_argument("--watch", help="Watch a directory for changes (includes initial scan)")
    parser.add_argument("--discover", help="DFS scan directory and process all supported images")
    parser.add_argument("--no-skip", action="store_true", help="Disable smart skip (reprocess all files)")
    # v3.2: --batch-size kept for backward compat but ignored (Phase pipeline handles batching)
    parser.add_argument("--batch-size", type=parse_batch_size, default='adaptive',
                       help="(Legacy, ignored) Batch size parameter")

    args = parser.parse_args()

    # ===== TIER COMPATIBILITY CHECK =====
    from backend.utils.tier_config import get_active_tier
    from backend.db.sqlite_client import SQLiteDB

    tier_name, tier_config = get_active_tier()
    visual_config = tier_config.get("visual", {})
    expected_dimension = visual_config.get("dimensions", 768)

    logger.info(f"[TIER CHECK] Current tier: {tier_name}, Expected dimension: {expected_dimension}")

    db = SQLiteDB()
    compat = db.check_tier_compatibility(tier_name, expected_dimension)

    if not compat['compatible']:
        from backend.utils.tier_compatibility import get_migration_steps

        logger.error("=" * 60)
        logger.error(f"[TIER CHANGE] {compat['message']}")
        logger.error("=" * 60)
        logger.error(f"  From: {compat['db_tier']} tier (dimension: {compat['db_dimension']})")
        logger.error(f"  To:   {compat['current_tier']} tier (dimension: {compat['current_dimension']})")
        logger.error("")
        logger.error(f"  Action: {compat['action'].upper()}")
        logger.error(f"  Reason: {compat['reason']}")
        logger.error("")

        if compat['user_prompt']:
            logger.error("Details:")
            for line in compat['user_prompt'].split('\n'):
                logger.error(f"  {line}")
            logger.error("")

        logger.error("Migration Steps:")
        steps = get_migration_steps(compat['db_tier'], compat['current_tier'])
        for step in steps:
            logger.error(f"  {step}")
        logger.error("")

        logger.error("OR revert config.yaml to previous tier:")
        logger.error(f"  ai_mode.override: {compat['db_tier']}")
        logger.error("=" * 60)

        logger.error("\nExecution blocked. Please migrate tier or revert config.")
        sys.exit(1)

    logger.info(f"[TIER CHECK] Compatibility OK")
    logger.info(f"  DB Tier: {compat['db_tier'] or 'empty'}")
    logger.info(f"  Current Tier: {compat['current_tier']}")
    logger.info(f"  Action: {compat['action']}")
    db.close()

    if args.discover:
        # v3.2: Always use Phase-based pipeline (ignore legacy batch_size)
        run_discovery(args.discover, skip_processed=not args.no_skip)
    elif args.files:
        # v3.4: Phase-based batch pipeline with smart skip
        import json
        try:
            file_list = json.loads(args.files)
            total = len(file_list)

            if total == 0:
                logger.info("[DONE] No files to process")
            else:
                # Smart skip: filter unchanged files (mtime + tier check)
                skip_db = SQLiteDB()
                to_process = []
                skipped = 0
                for fp_str in file_list:
                    fp = Path(fp_str)
                    if not args.no_skip and should_skip_file_enhanced(fp, skip_db, tier_name):
                        skipped += 1
                        logger.info(f"[SKIP] {fp.name} (unchanged, tier={tier_name})")
                    else:
                        to_process.append((fp, None, 0, []))
                skip_db.close()

                if skipped > 0:
                    logger.info(f"[SKIP] {skipped}/{total} files skipped (unchanged)")

                if len(to_process) == 0:
                    logger.info(f"[DONE] 0 processed, 0 errors (total: {total}, {skipped} skipped)")
                elif len(to_process) == 1:
                    logger.info(f"Processing: {to_process[0][0]}")
                    process_file(to_process[0][0])
                    logger.info(f"[DONE] 1 processed (total: {total}, {skipped} skipped)")
                else:
                    file_infos = to_process
                    stored, parse_errors, store_errors = process_batch_phased(file_infos)
                    logger.info(
                        f"[DONE] Batch complete: {stored} processed, "
                        f"{parse_errors + store_errors} errors (total: {total}, {skipped} skipped)"
                    )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
    elif args.file:
        process_file(Path(args.file))
    elif args.watch:
        start_watcher(args.watch)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
