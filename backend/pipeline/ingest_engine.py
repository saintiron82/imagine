"""
Ingest Engine - Main Entry Point for Data Pipeline.

Handles:
1. File detection (CLI or Watchdog)
2. Parser selection (Factory Pattern)
3. Execution and Result Logging

v3.5: Throughput-driven adaptive batch + NFC path normalization.
      Batch sizes start small (2) and grow/shrink based on memory pressure.
      Phase 1: Parse ALL (CPU parallel, no thumbnails) → Store metadata
      → Phase 2: VLM (adaptive sub-batch JIT load) → Store vision fields
      → Phase 3a: VV (adaptive sub-batch JIT load) → Unload SigLIP2
      → Phase 3b: MV (adaptive sub-batch, text-only) → Store vectors
"""

import argparse
import logging
import sys
import time
import io
import json as _json
import unicodedata
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
# SKIP when imported by worker_ipc — replacing stdout breaks the JSON IPC protocol
# and causes deadlock (two TextIOWrappers sharing the same buffer across threads)
if __name__ == "__main__" or "ingest_engine" in sys.argv[0:1]:
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

# Global DB handles (read connection + async write queue)
_global_sqlite_db = None
_global_db_writer = None


def _nfc(path) -> str:
    """Normalize file path to NFC form to prevent macOS NFD/NFC duplicates in DB."""
    return unicodedata.normalize('NFC', str(path).replace('\\', '/'))


def _ensure_sqlite_db():
    """Get or create process-wide SQLiteDB (read/metadata checks)."""
    from backend.db.sqlite_client import SQLiteDB
    global _global_sqlite_db
    if _global_sqlite_db is None:
        _global_sqlite_db = SQLiteDB()
        logger.info("SQLite database initialized")
    return _global_sqlite_db


def _get_db_writer():
    """Get or create process-wide async DB write queue."""
    from backend.db.write_queue import DBWriteQueue
    from backend.utils.config import get_config

    global _global_db_writer
    if _global_db_writer is not None:
        return _global_db_writer

    db = _ensure_sqlite_db()
    cfg = get_config()
    writer_cfg = cfg.get("batch_processing.db_writer", {})

    batch_size = writer_cfg.get("batch_size", 500) if isinstance(writer_cfg, dict) else 500
    flush_interval_s = writer_cfg.get("flush_interval_s", 0.15) if isinstance(writer_cfg, dict) else 0.15
    max_queue = writer_cfg.get("max_queue_size", 1024) if isinstance(writer_cfg, dict) else 1024

    _global_db_writer = DBWriteQueue(
        db_path=db.db_path,
        batch_size=batch_size,
        flush_interval_s=flush_interval_s,
        max_queue_size=max_queue,
    )
    return _global_db_writer


def _shutdown_db_writer():
    """Flush/close DB writer and read connection."""
    global _global_db_writer, _global_sqlite_db
    if _global_db_writer is not None:
        try:
            _global_db_writer.close()
        except Exception as e:
            logger.warning(f"DB writer shutdown failed: {e}")
        finally:
            _global_db_writer = None
    if _global_sqlite_db is not None:
        try:
            _global_sqlite_db.close()
        except Exception as e:
            logger.warning(f"SQLite DB shutdown failed: {e}")
        finally:
            _global_sqlite_db = None


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
    # Vector storage tracking (set by phase3a/3b incremental storage)
    stored_vv: bool = False
    stored_mv: bool = False


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

        # Compute content hash for path-independent identification
        from backend.utils.content_hash import compute_content_hash
        try:
            meta.content_hash = compute_content_hash(file_path)
        except Exception as e:
            logger.warning(f"content_hash computation failed: {e}")

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
        normalized = _nfc(file_path)
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

            # Generate embeddings from thumbnail
            embedding = None
            structure_embedding = None

            if thumb_path and thumb_path.exists():
                from PIL import Image
                import numpy as np

                try:
                    image = Image.open(thumb_path).convert("RGB")
                    
                    # SigLIP (VV)
                    embedding = _global_encoder.encode_image(image)
                    logger.debug(f"Generated VV embedding: {embedding.shape}")

                    # DINOv2 (Structure)
                    global _global_dinov2_encoder
                    if '_global_dinov2_encoder' not in globals() or _global_dinov2_encoder is None:
                        from backend.vector.dinov2_encoder import DinoV2Encoder
                        _global_dinov2_encoder = DinoV2Encoder()
                        logger.info(f"Structure encoder loaded: {_global_dinov2_encoder.model_name}")
                    
                    structure_embedding = _global_dinov2_encoder.encode_image(image)
                    logger.debug(f"Generated Structure embedding: {structure_embedding.shape}")

                except Exception as e:
                    logger.warning(f"Failed to generate embeddings: {e}")
                    if embedding is None:
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
                file_path=_nfc(file_path),
                metadata=meta.model_dump(),
                embedding=embedding,
                structure_embedding=structure_embedding
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
        thumb_img.close()
        return background
    rgb = thumb_img.convert("RGB")
    thumb_img.close()
    return rgb


def _load_visual_source_image(pf: ParsedFile) -> Optional["Image.Image"]:
    """
    Load an RGB image for vision/vector phases.

    Priority:
    1) Thumbnail file
    2) Original asset decode fallback (PSD/image)
    """
    if pf.thumb_path and pf.thumb_path.exists():
        return _load_and_composite_thumbnail(pf.thumb_path)

    try:
        from PIL import Image
        from backend.utils.config import get_config

        cfg = get_config()
        bg_color = cfg.get('thumbnail.index_composite_bg', '#FFFFFF')
        bg_rgb = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

        ext = pf.file_path.suffix.lower()
        if ext == '.psd':
            from psd_tools import PSDImage
            psd = PSDImage.open(pf.file_path)
            try:
                composite = psd.composite()
            except Exception as e:
                if "aggdraw" in str(e).lower():
                    logger.warning(
                        f"  [FALLBACK] PSD composite requires aggdraw, using embedded preview: {pf.file_path.name}"
                    )
                    if hasattr(psd, "topil"):
                        composite = psd.topil()
                    else:
                        logger.warning(
                            f"  [FALLBACK] PSDImage.topil() unavailable: {pf.file_path.name}"
                        )
                        composite = None
                else:
                    raise
            if composite is None:
                try:
                    import numpy as np
                    arr = psd.numpy()
                    if arr is not None and getattr(arr, "size", 0) > 0:
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr, 0, 255).astype(np.uint8)
                        if arr.ndim == 3 and arr.shape[2] in (3, 4):
                            composite = Image.fromarray(arr[:, :, :3], mode='RGB')
                            logger.warning(
                                f"  [FALLBACK] Using PSD numpy rasterization: {pf.file_path.name}"
                            )
                except Exception as e:
                    logger.warning(f"  [FALLBACK] PSD numpy rasterization failed: {pf.file_path.name}: {e}")
            if composite is None:
                return None
            if composite.mode == 'RGBA':
                background = Image.new('RGB', composite.size, bg_rgb)
                background.paste(composite, mask=composite.split()[-1])
                img = background
            else:
                img = composite.convert('RGB')
        else:
            with Image.open(pf.file_path) as raw:
                if raw.mode == 'RGBA':
                    background = Image.new('RGB', raw.size, bg_rgb)
                    background.paste(raw, mask=raw.split()[-1])
                    img = background
                else:
                    img = raw.convert('RGB')

        # Keep memory pressure bounded to tier thumbnail max-edge.
        max_edge = BaseParser.get_thumbnail_max_edge()
        if max(img.size) > max_edge:
            img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

        logger.info(f"  [FALLBACK] Using original decode for {pf.file_path.name}")
        return img
    except Exception as e:
        logger.warning(f"  [FALLBACK] Original decode failed for {pf.file_path.name}: {e}")
        return None


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
    normalized = _nfc(file_path)
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


def _build_synthetic_vision_result(pf: ParsedFile) -> dict:
    """
    Build deterministic fallback vision result when model inference is unavailable.
    Keeps pipeline forward-progress without blocking.
    """
    stem = pf.file_path.stem.replace("_", " ").replace("-", " ").strip()
    folder_hint = (pf.meta.folder_path or "").replace("/", " ").strip()
    base_caption = f"{stem} {folder_hint}".strip() or pf.file_path.name

    tags = []
    if pf.meta and pf.meta.semantic_tags:
        tags.extend([t for t in str(pf.meta.semantic_tags).split() if len(t) > 1][:8])
    if folder_hint:
        tags.extend([t for t in folder_hint.split() if len(t) > 1][:4])
    if not tags:
        tags = ["asset", "fallback"]

    # Deduplicate, preserve order
    seen = set()
    uniq_tags = []
    for t in tags:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq_tags.append(t)

    return {
        "caption": base_caption,
        "tags": uniq_tags[:12],
        "ocr": "",
        "color": "",
        "style": "",
        "image_type": "other",
        "art_style": None,
        "color_palette": None,
        "scene_type": None,
        "time_of_day": None,
        "weather": None,
        "character_type": None,
        "item_type": None,
        "ui_type": None,
    }


def _call_with_timeout(func, timeout_s: float):
    """
    Execute callable with a hard timeout on Unix main thread.

    Falls back to plain execution on unsupported environments.
    """
    if not timeout_s or timeout_s <= 0:
        return func()

    try:
        import signal
        import threading
        if sys.platform.startswith("win") or threading.current_thread() is not threading.main_thread():
            return func()
    except Exception:
        return func()

    class _TimedOut(Exception):
        pass

    def _handler(_signum, _frame):
        raise _TimedOut()

    prev = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return func()
    except _TimedOut as e:
        raise TimeoutError(f"Vision inference timed out after {timeout_s:.1f}s") from e
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _load_vlm_model_with_policy(analyzer, preload_timeout_s: float):
    """
    Load VLM model with platform-aware timeout policy.

    On macOS MPS, hard signal timeouts can interrupt heavy tensor materialization
    and produce false timeout failures despite healthy progress. In that path,
    preload timeout is treated as a warning threshold only.
    """
    device = getattr(analyzer, "device", None)
    is_macos_mps = (sys.platform == "darwin" and device == "mps")

    if not is_macos_mps:
        _call_with_timeout(lambda: analyzer._load_model(), preload_timeout_s)
        return

    if preload_timeout_s and preload_timeout_s > 0:
        logger.info(
            "[VLM preload] macOS+MPS detected; hard preload timeout is disabled "
            f"(warning threshold={preload_timeout_s:.1f}s)."
        )
    t0 = time.perf_counter()
    analyzer._load_model()
    elapsed = time.perf_counter() - t0
    if preload_timeout_s and preload_timeout_s > 0 and elapsed > preload_timeout_s:
        logger.warning(
            f"[WARN:VLM] Model preload took {elapsed:.1f}s "
            f"(threshold {preload_timeout_s:.1f}s) on macOS+MPS."
        )


def _prewarm_vlm_imports() -> None:
    """
    Pre-import Qwen3-VL transformer classes on the main thread.

    This avoids late heavy import chains being triggered after threaded parse work,
    which can appear as a freeze before VLM weight loading starts.
    """
    try:
        from backend.utils.config import get_config
        from backend.utils.tier_config import get_active_tier

        cfg = get_config()
        if not bool(cfg.get("vision.prewarm_imports", True)):
            return

        _, tier_cfg = get_active_tier()
        model_id = str(tier_cfg.get("vlm", {}).get("model", "") or "")
        if "Qwen3-VL" not in model_id and "Qwen3VL" not in model_id:
            return

        t0 = time.perf_counter()
        logger.info("[VLM prewarm] Importing Qwen3-VL transformer classes on main thread...")
        from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: F401
        logger.info(f"[VLM prewarm] Import complete in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"[VLM prewarm] skipped: {e}")


def _run_vlm_inference_with_policy(analyzer, infer_fn, timeout_s: float):
    """
    Run VLM inference with platform-aware timeout policy.

    On macOS+MPS, hard SIGALRM timeouts can fail to interrupt long-running
    kernel sections predictably. In that path, timeout is treated as a warning
    threshold while generation-side limits still bound output length.
    """
    device = getattr(analyzer, "device", None)
    is_macos_mps = (sys.platform == "darwin" and device == "mps")

    if not is_macos_mps:
        return _call_with_timeout(infer_fn, timeout_s)

    if timeout_s and timeout_s > 0:
        logger.info(
            "[VLM inference] macOS+MPS detected; hard inference timeout is disabled "
            f"(warning threshold={timeout_s:.1f}s)."
        )

    t0 = time.perf_counter()
    out = infer_fn()
    elapsed = time.perf_counter() - t0
    if timeout_s and timeout_s > 0 and elapsed > timeout_s:
        logger.warning(
            f"[WARN:VLM] Inference took {elapsed:.1f}s "
            f"(threshold {timeout_s:.1f}s) on macOS+MPS."
        )
    return out


def _derive_structure_from_vv(vv_vec, target_dim: int = 768):
    """
    Deterministic fallback: derive a structure vector from VV when DINOv2 is unavailable.
    """
    import numpy as np

    if vv_vec is None:
        return None
    arr = np.asarray(vv_vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None

    if arr.size >= target_dim:
        out = arr[:target_dim].copy()
    else:
        out = np.zeros(target_dim, dtype=np.float32)
        out[:arr.size] = arr

    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out.astype(np.float32)


def _check_thumbnail_size(thumb_path: Path, min_edge: int) -> bool:
    """Return True if thumbnail meets minimum size requirement."""
    if not thumb_path or not thumb_path.exists():
        return False
    try:
        from PIL import Image
        with Image.open(thumb_path) as img:
            w, h = img.size
            return max(w, h) >= min_edge
    except Exception:
        return False


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

    # Compute content hash for path-independent identification
    from backend.utils.content_hash import compute_content_hash
    try:
        meta.content_hash = compute_content_hash(fp)
    except Exception as e:
        logger.warning(f"content_hash failed for {fp.name}: {e}")

    # Resolve thumbnail (path only, no image loading — JIT in Phase 2/3)
    pf.thumb_path = _resolve_thumb_path(meta)

    # Ensure thumbnail exists and meets tier size requirement.
    # Missing/invalid thumbnails create an infinite "partially processed" loop
    # because VV/Structure phases cannot run without an image source.
    required_edge = BaseParser.get_thumbnail_max_edge()
    needs_regen = False
    regen_reason = ""
    if not pf.thumb_path or not pf.thumb_path.exists():
        needs_regen = True
        regen_reason = "missing"
    elif not _check_thumbnail_size(pf.thumb_path, required_edge):
        needs_regen = True
        regen_reason = "undersized"

    if needs_regen:
        logger.info(
            f"  [REGEN] {fp.name}: thumbnail {regen_reason}, regenerating at {required_edge}px"
        )
        try:
            new_thumb = parser._create_thumbnail(
                *(_open_for_thumbnail(fp, parser))
            )
            if new_thumb:
                pf.thumb_path = new_thumb
                meta.thumbnail_url = str(new_thumb)
                meta.visual_source_path = str(new_thumb)
            else:
                logger.warning(f"  [REGEN] thumbnail regeneration returned empty path: {fp.name}")
        except Exception as e:
            logger.warning(f"  [REGEN] thumbnail regeneration failed for {fp.name}: {e}")

    # Build MC.raw context
    pf.mc_raw = _build_mc_raw(meta)

    # Set tier metadata
    _set_tier_metadata(meta)

    # Normalize paths
    _normalize_paths(meta, fp)

    return pf


def _open_for_thumbnail(fp: Path, parser):
    """Open file source for thumbnail regeneration, returning args for parser._create_thumbnail()."""
    from PIL import Image
    if hasattr(parser, '__class__') and parser.__class__.__name__ == 'PSDParser':
        from psd_tools import PSDImage
        psd = PSDImage.open(fp)
        return psd, fp
    else:
        img = Image.open(fp)
        return img, fp


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


def phase2_vision_adaptive(
    parsed_files: List[ParsedFile], controller, monitor,
    progress_callback=None, phase_progress_callback=None
) -> None:
    """Phase 2: VLM analysis with adaptive batch sizing, JIT thumbnails, and incremental DB storage."""
    import gc
    from backend.utils.config import get_config
    writer = _get_db_writer()
    write_jobs = []
    cfg = get_config()
    vlm_timeout_s = float(cfg.get("vision.vlm_timeout_s", 45.0) or 45.0)
    vlm_preload_timeout_s = float(
        cfg.get("vision.vlm_preload_timeout_s", max(180.0, vlm_timeout_s * 3.0))
        or max(180.0, vlm_timeout_s * 3.0)
    )
    vlm_fail_fast = bool(cfg.get("vision.vlm_fail_fast", True))

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
        # Allow fallback decode from original file when thumbnail is missing.
        vision_indices.append(idx)

    if skipped > 0:
        logger.info(f"  [SKIP:Vision] {skipped} files skipped (MC unchanged)")

    if not vision_indices:
        logger.info("STEP 2/4 AI Vision — no images to analyze")
        return

    t0 = time.perf_counter()
    total_vision = len(vision_indices)
    logger.info(f"STEP 2/4 AI Vision ({total_vision} images, adaptive batch)")

    from backend.vision.vision_factory import get_vision_analyzer
    analyzer = get_vision_analyzer()
    analyzer_device = getattr(analyzer, "device", None)
    force_vlm_single = analyzer_device == "cpu"
    if force_vlm_single:
        if analyzer_device == "cpu":
            logger.warning(
                "VLM is running on CPU; forcing sub-batch size to 1 for responsive progress."
            )
    vlm_available = True
    if hasattr(analyzer, "_load_model"):
        try:
            _load_vlm_model_with_policy(analyzer, vlm_preload_timeout_s)
        except TimeoutError:
            msg = (
                f"[TIMEOUT:VLM] Model load exceeded {vlm_preload_timeout_s:.1f}s; "
                "disabling VLM for remaining files in this run."
            )
            if vlm_fail_fast:
                logger.error(f"{msg} (fail-fast enabled)")
                raise RuntimeError(
                    f"VLM preload timeout ({vlm_preload_timeout_s:.1f}s). "
                    "Current run is aborted. Check device acceleration and model load path."
                )
            vlm_available = False
            logger.warning(msg)
        except Exception as e:
            msg = (
                f"[FAIL:VLM] Model preload failed ({e}); "
                "disabling VLM for remaining files in this run."
            )
            if vlm_fail_fast:
                logger.error(f"{msg} (fail-fast enabled)")
                raise RuntimeError(
                    f"VLM preload failed: {e}. "
                    "Current run is aborted. Resolve model/device issue and retry."
                )
            vlm_available = False
            logger.warning(msg)

    monitor.set_baseline()
    processed = 0

    while processed < total_vision:
        batch_size = 1 if force_vlm_single else controller.get_batch_size('vlm')
        chunk = vision_indices[processed:processed + batch_size]
        t_sub = time.perf_counter()

        # JIT: Load vision source for this sub-batch only
        for idx in chunk:
            pf = parsed_files[idx]
            try:
                pf.thumb_image_rgb = _load_visual_source_image(pf)
            except Exception as e:
                logger.warning(f"Visual source load failed for {pf.file_path.name}: {e}")
            _compute_dhash(pf)

        # Build vision items for this sub-batch
        vision_items = []
        vision_results = []
        valid_chunk = []
        for idx in chunk:
            pf = parsed_files[idx]
            if pf.thumb_image_rgb is not None:
                vision_items.append((pf.thumb_image_rgb, pf.mc_raw))
                valid_chunk.append(idx)

        if vision_items:
            # Progress callback
            def _progress(i, total, result, _offset=processed):
                pf_inner = parsed_files[valid_chunk[i]]
                logger.info(f"  [{_offset + i + 1}/{total_vision}] {pf_inner.file_path.name} → {result.get('image_type', '?')}")
                if progress_callback:
                    progress_callback(_offset + i, total_vision, result)

            if not vlm_available:
                vision_results = []
                for i, _ in enumerate(vision_items):
                    pf_fb = parsed_files[valid_chunk[i]]
                    r = _build_synthetic_vision_result(pf_fb)
                    vision_results.append(r)
                    _progress(i, len(vision_items), r)
            else:
                try:
                    if len(vision_items) == 1:
                        logger.info(
                            f"  [RUN:VLM] {processed + 1}/{total_vision} "
                            f"{parsed_files[valid_chunk[0]].file_path.name}"
                        )
                    else:
                        logger.info(
                            f"  [RUN:VLM] chunk {processed + 1}-"
                            f"{processed + len(vision_items)}/{total_vision} "
                            f"(batch={len(vision_items)})"
                        )
                    if hasattr(analyzer, 'classify_and_analyze_sequence'):
                        vision_results = _run_vlm_inference_with_policy(
                            analyzer,
                            lambda: analyzer.classify_and_analyze_sequence(
                                vision_items, progress_callback=_progress
                            ),
                            vlm_timeout_s,
                        )
                    else:
                        vision_results = _run_vlm_inference_with_policy(
                            analyzer,
                            lambda: [
                                analyzer.classify_and_analyze(img, context=ctx)
                                for (img, ctx) in vision_items
                            ],
                            vlm_timeout_s,
                        )
                        for i, r in enumerate(vision_results):
                            _progress(i, len(vision_items), r)
                except TimeoutError:
                    msg = (
                        f"  [TIMEOUT:VLM] {len(vision_items)} images exceeded "
                        f"{vlm_timeout_s:.1f}s; switching to synthetic fallback "
                        "for remaining files in this run"
                    )
                    if vlm_fail_fast:
                        logger.error(f"{msg} (fail-fast enabled)")
                        raise RuntimeError(
                            f"VLM inference timeout ({vlm_timeout_s:.1f}s) at "
                            f"{processed + 1}/{total_vision}. Run aborted."
                        )
                    vlm_available = False
                    logger.warning(msg)
                    vision_results = []
                    for i, _ in enumerate(vision_items):
                        pf_fb = parsed_files[valid_chunk[i]]
                        r = _build_synthetic_vision_result(pf_fb)
                        vision_results.append(r)
                        _progress(i, len(vision_items), r)
                except Exception as e:
                    msg = (
                        f"  [FAIL:VLM] batch failed ({e}); switching to synthetic fallback "
                        "for remaining files in this run"
                    )
                    if vlm_fail_fast:
                        logger.error(f"{msg} (fail-fast enabled)")
                        raise RuntimeError(
                            f"VLM inference failed at {processed + 1}/{total_vision}: {e}. "
                            "Run aborted."
                        )
                    vlm_available = False
                    logger.warning(msg)
                    vision_results = []
                    for i, _ in enumerate(vision_items):
                        pf_fb = parsed_files[valid_chunk[i]]
                        r = _build_synthetic_vision_result(pf_fb)
                        vision_results.append(r)
                        _progress(i, len(vision_items), r)

            # Apply results
            for i, result in enumerate(vision_results):
                pf = parsed_files[valid_chunk[i]]
                _apply_vision_result(pf.meta, result)

        # Release thumbnails and vision intermediates immediately
        del vision_items, vision_results
        for idx in chunk:
            parsed_files[idx].thumb_image_rgb = None

        # Incremental DB storage: enqueue vision writes per sub-batch
        if valid_chunk and writer:
            for idx in valid_chunk:
                pf = parsed_files[idx]
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
                    fut = writer.submit_vision(_nfc(pf.file_path), fields)
                    write_jobs.append((pf, fut))
                except Exception as e:
                    logger.error(f"  [FAIL:vision-store] {pf.file_path.name}: {e}")
        # Release VLM input context for ALL items (mc_caption/ai_tags kept for Phase 3b MV)
        for idx in chunk:
            parsed_files[idx].mc_raw = None

        processed += len(chunk)
        gc.collect()
        elapsed_sub = time.perf_counter() - t_sub

        # Adaptive: decide next batch size (throughput-driven)
        decision = None
        if not force_vlm_single:
            decision = controller.after_sub_batch('vlm', len(chunk), elapsed_sub)

        # Emit phase progress
        if phase_progress_callback:
            phase_progress_callback(processed, batch_size)

        if decision and decision.abort:
            logger.error(f"[ADAPTIVE:vlm] Aborting VLM phase at {processed}/{total_vision}")
            break

    elapsed = time.perf_counter() - t0
    if write_jobs:
        stored_ok = 0
        stored_fail = 0
        for pf, fut in write_jobs:
            try:
                fut.result()
                stored_ok += 1
            except Exception as e:
                stored_fail += 1
                logger.error(f"  [FAIL:vision-store] {pf.file_path.name}: {e}")
        if stored_fail > 0:
            logger.warning(f"  [VISION-STORE] {stored_ok}/{len(write_jobs)} saved")
    logger.info(f"STEP 2/4 completed in {elapsed:.1f}s ({processed}/{total_vision})")


def phase3a_vv_adaptive(
    parsed_files: List[ParsedFile], controller, monitor,
    phase_progress_callback=None
) -> dict:
    """
    Phase 3a: SigLIP2 VV encoding with adaptive batch sizing and incremental DB storage.

    Returns empty dict (vectors saved to DB immediately per sub-batch).
    """
    import gc
    import numpy as np
    writer = _get_db_writer()
    write_jobs = []

    valid_indices = [i for i, pf in enumerate(parsed_files) if pf.error is None]

    vv_indices = []
    vv_skipped = 0
    for i in valid_indices:
        pf = parsed_files[i]
        if pf.skip_embed_vv:
            vv_skipped += 1
            continue
        # Allow fallback decode from original file when thumbnail is missing.
        vv_indices.append(i)
    if vv_skipped > 0:
        logger.info(f"  [SKIP:VV] {vv_skipped} files skipped (same SigLIP2 model)")

    vv_results = {}

    if not vv_indices:
        logger.info("  VV: no images to encode")
        return vv_results

    t0 = time.perf_counter()
    logger.info(f"STEP 3a/4 VV Embedding ({len(vv_indices)} images, adaptive batch)")

    from backend.vector.siglip2_encoder import SigLIP2Encoder
    global _global_encoder
    if '_global_encoder' not in globals() or _global_encoder is None:
        _global_encoder = SigLIP2Encoder()
        logger.info(f"SigLIP2 encoder loaded: {_global_encoder.model_name}")

    # Let SigLIP2 discover optimal batch, use as controller max
    # (This runs on first encode_image_batch call automatically)

    monitor.set_baseline()
    processed = 0
    total_vv = len(vv_indices)

    while processed < total_vv:
        batch_size = controller.get_batch_size('vv')
        chunk = vv_indices[processed:processed + batch_size]
        t_sub = time.perf_counter()

        # JIT: Load visual source from thumbnail or original fallback
        images = []
        chunk_valid = []
        for idx in chunk:
            pf = parsed_files[idx]
            try:
                img = _load_visual_source_image(pf)
                if img is not None:
                    images.append(img)
                    chunk_valid.append(idx)
            except Exception as e:
                logger.warning(f"VV visual source load failed for {pf.file_path.name}: {e}")

        if images:
            vv_vectors = _global_encoder.encode_image_batch(images)

            # Lazy load DINOv2 for Structure vectors
            global _global_dinov2_encoder
            if '_global_dinov2_encoder' not in globals() or _global_dinov2_encoder is None:
                from backend.vector.dinov2_encoder import DinoV2Encoder
                _global_dinov2_encoder = DinoV2Encoder()
                logger.info(f"Structure encoder loaded: {_global_dinov2_encoder.model_name}")

            # Compute DINOv2 vectors
            dinov2_vectors = []
            for img in images:
                try:
                    dinov2_vectors.append(_global_dinov2_encoder.encode_image(img))
                except Exception:
                    dinov2_vectors.append(None)

            # Incremental DB storage: enqueue VV + Structure vector writes
            for j, vec in enumerate(vv_vectors):
                idx = chunk_valid[j]
                pf = parsed_files[idx]
                dino_vec = dinov2_vectors[j] if j < len(dinov2_vectors) else None

                if pf.db_file_id and vec is not None and writer:
                    try:
                        fut = writer.submit_vectors(
                            pf.db_file_id,
                            vv_vec=vec,
                            mv_vec=None,
                            structure_vec=dino_vec,
                        )
                        write_jobs.append((pf, fut))
                    except Exception as e:
                        logger.error(f"  [FAIL:vv-store] {pf.file_path.name}: {e}")
            del images, vv_vectors, dinov2_vectors  # Release thumbnails + vectors

        processed += len(chunk)
        gc.collect()
        elapsed_sub = time.perf_counter() - t_sub

        # Adaptive decision (throughput-driven)
        decision = controller.after_sub_batch('vv', len(chunk), elapsed_sub)

        if phase_progress_callback:
            phase_progress_callback(processed, 'vv', batch_size)

        if decision.abort:
            logger.error(f"[ADAPTIVE:vv] Aborting VV phase at {processed}/{total_vv}")
            break

    elapsed = time.perf_counter() - t0
    if write_jobs:
        stored_ok = 0
        stored_fail = 0
        for pf, fut in write_jobs:
            try:
                fut.result()
                pf.stored_vv = True
                stored_ok += 1
            except Exception as e:
                stored_fail += 1
                logger.error(f"  [FAIL:vv-store] {pf.file_path.name}: {e}")
        if stored_fail > 0:
            logger.warning(f"  [VV-STORE] {stored_ok}/{len(write_jobs)} saved")
    logger.info(f"STEP 3a/4 VV completed in {elapsed:.1f}s ({processed}/{total_vv} encoded + stored)")


def phase3b_mv_adaptive(
    parsed_files: List[ParsedFile], controller, monitor,
    phase_progress_callback=None
) -> dict:
    """
    Phase 3b: Qwen3-Embedding MV encoding with adaptive batch sizing and incremental DB storage.

    Returns empty dict (vectors saved to DB immediately per sub-batch).
    """
    import gc
    writer = _get_db_writer()
    write_jobs = []

    from backend.utils.config import get_config
    cfg = get_config()
    text_enabled = cfg.get("embedding.text.enabled", False)

    mv_results = {}

    if not text_enabled:
        logger.info("  MV: text embedding disabled")
        return mv_results

    valid_indices = [i for i, pf in enumerate(parsed_files) if pf.error is None]

    # Collect texts to encode
    from backend.vector.text_embedding import get_text_embedding_provider, build_document_text

    global _global_text_provider
    if '_global_text_provider' not in globals() or _global_text_provider is None:
        _global_text_provider = get_text_embedding_provider()

    mv_items = []  # (file_index, text)
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
            mv_items.append((i, doc_text))

    if mv_skipped > 0:
        logger.info(f"  [SKIP:MV] {mv_skipped} files skipped (same text embed model)")

    if not mv_items:
        logger.info("  MV: no texts to encode")
        return mv_results

    t0 = time.perf_counter()
    total_mv = len(mv_items)
    logger.info(f"STEP 3b/4 MV Embedding ({total_mv} texts, adaptive batch)")

    monitor.set_baseline()
    processed = 0

    while processed < total_mv:
        batch_size = controller.get_batch_size('mv')
        chunk = mv_items[processed:processed + batch_size]
        t_sub = time.perf_counter()

        texts = [text for _, text in chunk]
        encoded_pairs = []  # (file_idx, vec) for incremental storage
        try:
            if hasattr(_global_text_provider, 'encode_batch'):
                vecs = _global_text_provider.encode_batch(texts)
            else:
                vecs = [_global_text_provider.encode(t) for t in texts]

            for j, vec in enumerate(vecs):
                file_idx = chunk[j][0]
                encoded_pairs.append((file_idx, vec))
        except Exception as e:
            logger.warning(f"  MV batch encoding failed: {e}")
            # Fallback: individual encoding
            for j, (file_idx, text) in enumerate(chunk):
                try:
                    vec = _global_text_provider.encode(text)
                    encoded_pairs.append((file_idx, vec))
                except Exception:
                    pass

        # Incremental DB storage: enqueue MV vector writes
        if encoded_pairs and writer:
            for file_idx, vec in encoded_pairs:
                pf = parsed_files[file_idx]
                if not pf.db_file_id:
                    logger.warning(f"  [SKIP:mv-store] {pf.file_path.name}: no db_file_id")
                    continue
                if vec is None:
                    logger.warning(f"  [SKIP:mv-store] {pf.file_path.name}: vec is None")
                    continue
                try:
                    fut = writer.submit_vectors(pf.db_file_id, vv_vec=None, mv_vec=vec)
                    write_jobs.append((pf, fut))
                except Exception as e:
                    logger.error(f"  [FAIL:mv-store] {pf.file_path.name}: {e}")
            del encoded_pairs

        processed += len(chunk)
        gc.collect()
        elapsed_sub = time.perf_counter() - t_sub

        decision = controller.after_sub_batch('mv', len(chunk), elapsed_sub)

        if phase_progress_callback:
            phase_progress_callback(processed, 'mv', batch_size)

        if decision.abort:
            logger.error(f"[ADAPTIVE:mv] Aborting MV phase at {processed}/{total_mv}")
            break

    elapsed = time.perf_counter() - t0
    if write_jobs:
        stored_ok = 0
        stored_fail = 0
        for pf, fut in write_jobs:
            try:
                fut.result()
                pf.stored_mv = True
                stored_ok += 1
            except Exception as e:
                stored_fail += 1
                logger.error(f"  [FAIL:mv-store] {pf.file_path.name}: {e}")
        if stored_fail > 0:
            logger.warning(f"  [MV-STORE] {stored_ok}/{len(write_jobs)} saved")
    logger.info(f"STEP 3b/4 MV completed in {elapsed:.1f}s ({processed}/{total_mv} encoded + stored)")


def phase_store_metadata(parsed_files: List[ParsedFile]) -> int:
    """Phase 1 storage: INSERT basic metadata + save JSON files."""
    _ensure_sqlite_db()
    writer = _get_db_writer()

    stored = 0
    pending = []
    for pf in parsed_files:
        if pf.error is not None:
            continue
        try:
            pf.parser._save_json(pf.meta, pf.file_path)
            fut = writer.submit_metadata(
                file_path=_nfc(pf.file_path),
                metadata=pf.meta.model_dump()
            )
            pending.append((pf, fut))
        except Exception as e:
            logger.error(f"  [FAIL:meta] {pf.file_path.name}: {e}")
            pf.error = f"Metadata store failed: {e}"

    for pf, fut in pending:
        try:
            pf.db_file_id = fut.result()
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
            _global_sqlite_db.update_vision_fields(_nfc(pf.file_path), fields)
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
    from backend.utils.config import get_config as _get_cfg
    from backend.utils.tier_config import get_active_tier

    global _global_sqlite_db
    if '_global_sqlite_db' not in globals() or _global_sqlite_db is None:
        _global_sqlite_db = SQLiteDB()

    _, tier_config = get_active_tier()
    current_vlm = tier_config.get("vlm", {}).get("model", "")
    current_vv_model = tier_config.get("visual", {}).get("model", "")
    current_mv_model = tier_config.get("text_embed", {}).get("model", "")
    cfg = _get_cfg()
    verify_hash = bool(cfg.get("batch_processing.skip_verify_hash", False))

    skip_v, skip_vv, skip_mv = 0, 0, 0

    for pf in parsed_files:
        if pf.error is not None:
            continue

        info = _global_sqlite_db.get_file_phase_info(_nfc(pf.file_path.resolve()))
        if info is None:
            continue  # New file, no skip possible

        # mtime check: if file content changed, run ALL phases
        stat = pf.file_path.stat()
        stored_mtime = info.get("modified_at")
        if stored_mtime:
            current_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
            if stored_mtime.replace('T', ' ') != current_mtime.replace('T', ' '):
                continue  # File changed → run all phases

        stored_size = info.get("file_size")
        if stored_size is not None and int(stored_size) != int(stat.st_size):
            continue  # File size changed

        stored_hash = info.get("content_hash")
        current_hash = getattr(pf.meta, "content_hash", None)
        if verify_hash and stored_hash and current_hash and stored_hash != current_hash:
            continue  # Content hash changed

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

        # VV: skip if vector exists with same SigLIP2 model AND structure exists
        # If structure is missing (has_structure=False), we must not skip Phase 3a.
        # Phase 3a handles both VV and Structure generation.
        has_structure = info.get("has_structure", False)
        if info["has_vv"] and info.get("embedding_model") == current_vv_model and has_structure:
            pf.skip_embed_vv = True
            skip_vv += 1
        elif info["has_vv"] and info.get("embedding_model") == current_vv_model and not has_structure:
             # Partial skip? No, ingest engine Phases are coarse. 
             # If we enter Phase 3a, it will regenerate VV unless we add fine-grained logic there.
             # But generating VV (SigLIP) is fast. It's safer to just let it run or modify Phase 3a.
             # For now, we DON'T skip, so Phase 3a runs. 
             # Phase 3a verifies if VV is needed? No, it just overwrites. That's fine.
             pass

        # MV: skip if vector exists with same text embed model
        if info["has_mv"] and info.get("text_embed_model") == current_mv_model:
            pf.skip_embed_mv = True
            skip_mv += 1

    if skip_v > 0 or skip_vv > 0 or skip_mv > 0:
        logger.info(
            f"[SKIP:Phase] Vision: {skip_v}, VV: {skip_vv}, MV: {skip_mv} "
            f"files can skip (model+mtime match)"
        )


def _lighten_parsed_files(parsed_files: List[ParsedFile]):
    """
    Release heavy fields from ParsedFile after Phase 1 metadata storage.

    Frees parser instances (which hold PSDImage circular references),
    layer_tree dicts, and raw metadata dicts that are already in DB.
    """
    import gc
    for pf in parsed_files:
        if pf.error is not None:
            continue
        # Parser holds PSDImage with circular refs (~2.75GB for 100 PSD files)
        pf.parser = None
        # layer_tree is already saved to DB/JSON
        if pf.meta and hasattr(pf.meta, 'layer_tree') and pf.meta.layer_tree:
            pf.meta.layer_tree = {}
        # Raw metadata dict is already in DB
        if pf.meta and hasattr(pf.meta, 'metadata') and pf.meta.metadata:
            pf.meta.metadata = {}
    # 2-pass GC needed for circular reference collection
    gc.collect()
    gc.collect()


def _force_memory_reclaim(monitor, label: str):
    """GC + MPS/CUDA cache clear with verification (up to 3 retries)."""
    import gc
    try:
        import torch
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_mps = False
        has_cuda = False

    before = monitor.snapshot()

    for attempt in range(3):
        gc.collect()
        gc.collect()
        if has_mps:
            try:
                torch.mps.synchronize()  # Complete async ops BEFORE freeing
            except Exception:
                pass
            torch.mps.empty_cache()
        elif has_cuda:
            torch.cuda.empty_cache()

        after = monitor.snapshot()
        dropped = before.rss_gb - after.rss_gb

        if dropped > 0 or after.pressure_ratio < 0.65:
            logger.info(f"[MEM:{label}] Reclaimed {dropped:.1f}GB (attempt {attempt + 1})")
            break

        time.sleep(0.5)  # MPS async deallocation delay
    else:
        after = monitor.snapshot()
        logger.warning(
            f"[MEM:{label}] Memory not fully reclaimed after 3 attempts "
            f"(RSS: {before.rss_gb:.1f}→{after.rss_gb:.1f}GB)"
        )

    monitor.log_status(label)


def _unload_vlm_verified(monitor):
    """Unload VLM and verify memory drop."""
    try:
        from backend.vision.vision_factory import VisionAnalyzerFactory
        # Access cached instance directly — don't call get_vision_analyzer()
        # which would create a NEW instance if cache is already None
        VisionAnalyzerFactory.reset()  # reset() now calls unload_model() internally
    except Exception:
        pass

    global _global_vision_analyzer
    if '_global_vision_analyzer' in globals():
        _global_vision_analyzer = None

    _force_memory_reclaim(monitor, "vlm_unload")


def _unload_siglip2_verified(monitor):
    """Unload SigLIP2 and verify memory drop."""
    global _global_encoder
    if '_global_encoder' in globals() and _global_encoder is not None:
        _global_encoder.unload()
        _global_encoder = None

    _force_memory_reclaim(monitor, "siglip2_unload")


def _unload_mv_verified(monitor):
    """Unload MV model and verify memory drop."""
    global _global_text_provider
    if '_global_text_provider' in globals() and _global_text_provider is not None:
        if hasattr(_global_text_provider, 'unload'):
            _global_text_provider.unload()
        _global_text_provider = None

    # Also reset module-level singleton
    try:
        from backend.vector.text_embedding import reset_provider
        reset_provider()
    except Exception:
        pass

    _force_memory_reclaim(monitor, "mv_unload")


def process_batch_phased(
    file_infos: List[tuple],
    vv_batch: int = None,
    mv_batch: int = None,
    parse_workers: int = 4,
    progress_callback=None,
    allow_phase_skip: bool = True,
) -> Tuple[int, int, int]:
    """
    Adaptive batch pipeline v3.4: memory-aware Phase separation.

    Batch sizes start small and grow/shrink based on runtime memory pressure.
    Models are loaded one at a time and fully unloaded between phases.

    Phase 1: Parse ALL (CPU parallel) → Store metadata → Lighten ParsedFiles
    Phase 2: VLM (adaptive batch) → Store vision → Unload VLM
    Phase 3a: VV (adaptive batch) → Unload SigLIP2
    Phase 3b: MV (adaptive batch) → Unload Qwen3-Embedding
    Phase 4: Store vectors

    Returns:
        (stored, parse_errors, store_errors) counts
    """
    import gc
    import numpy as np
    from backend.utils.memory_monitor import MemoryMonitor
    from backend.utils.adaptive_batch_controller import AdaptiveBatchController
    from backend.utils.config import get_config

    t_total = time.perf_counter()
    total = len(file_infos)

    # Load adaptive config
    cfg = get_config()
    adaptive_cfg = cfg.get("batch_processing.adaptive", {})
    memory_budget = adaptive_cfg.get("memory_budget_gb", 20.0) if isinstance(adaptive_cfg, dict) else 20.0

    # Initialize monitor + controller
    monitor = MemoryMonitor.get_instance()
    phase_limits = {}
    if isinstance(adaptive_cfg, dict):
        if adaptive_cfg.get("vlm_initial_batch"):
            phase_limits['vlm'] = {'initial': adaptive_cfg['vlm_initial_batch']}
        if adaptive_cfg.get("vv_initial_batch"):
            phase_limits['vv'] = {'initial': adaptive_cfg['vv_initial_batch']}
        if adaptive_cfg.get("mv_initial_batch"):
            phase_limits['mv'] = {'initial': adaptive_cfg['mv_initial_batch']}

    controller = AdaptiveBatchController(
        phase_limits=phase_limits,
        memory_budget_gb=memory_budget,
    )

    monitor.set_baseline()
    monitor.log_status("pipeline_start")

    logger.info(f"[BATCH] Adaptive pipeline v3.5: {total} files (throughput-driven batch sizing)")
    _prewarm_vlm_imports()

    # Cumulative phase counters (emitted as [PHASE] for frontend tracking)
    # P=Parse, MC=Meta-Context Caption, VV=Visual Vector, MV=Meaning Vector
    cum_parse = 0
    cum_mc = 0
    cum_vv = 0
    cum_mv = 0
    active_batch_info = ""

    def _emit_phase_progress():
        logger.info(
            f"[PHASE] P:{cum_parse} MC:{cum_mc} VV:{cum_vv} MV:{cum_mv} "
            f"T:{total} {active_batch_info}"
        )

    # ── Phase 1: Parse ALL (CPU parallel, metadata only) ──
    parsed = phase1_parse_all(file_infos, max_workers=parse_workers)
    valid_count = sum(1 for pf in parsed if pf.error is None)
    cum_parse = valid_count
    _emit_phase_progress()

    # Store Phase 1: metadata + JSON
    phase_store_metadata(parsed)

    # Lighten ParsedFiles: release parser/layer_tree/metadata
    _lighten_parsed_files(parsed)
    monitor.log_status("after_phase1_lighten")

    # Per-phase smart skip (can be disabled by --no-skip)
    if allow_phase_skip:
        _check_phase_skip(parsed)
    else:
        logger.info("[SKIP:Phase] disabled (--no-skip): forcing MC/VV/MV reprocessing")

    # ── Phase 2: VLM (adaptive batch) → MC generation ──
    def _on_vision_progress(count, batch_size=2):
        nonlocal cum_mc, active_batch_info
        cum_mc = count
        active_batch_info = f"B:{batch_size}"
        _emit_phase_progress()

    phase2_vision_adaptive(
        parsed, controller, monitor,
        progress_callback=progress_callback,
        phase_progress_callback=_on_vision_progress,
    )
    cum_mc = valid_count
    active_batch_info = ""
    _emit_phase_progress()

    # Phase 2 vision fields already saved incrementally per sub-batch

    # Unload VLM completely + verify
    _unload_vlm_verified(monitor)

    # ── Phase 3a: VV (adaptive batch) ──
    def _on_vv_progress(count, kind, batch_size=4):
        nonlocal cum_vv, active_batch_info
        cum_vv = count
        active_batch_info = f"B:{batch_size}"
        _emit_phase_progress()

    phase3a_vv_adaptive(
        parsed, controller, monitor,
        phase_progress_callback=_on_vv_progress,
    )
    cum_vv = valid_count
    active_batch_info = ""
    _emit_phase_progress()

    # Unload SigLIP2 completely + verify
    _unload_siglip2_verified(monitor)

    # ── Phase 3b: MV (adaptive batch) ──
    def _on_mv_progress(count, kind, batch_size=16):
        nonlocal cum_mv, active_batch_info
        cum_mv = count
        active_batch_info = f"B:{batch_size}"
        _emit_phase_progress()

    phase3b_mv_adaptive(
        parsed, controller, monitor,
        phase_progress_callback=_on_mv_progress,
    )
    cum_mv = valid_count
    active_batch_info = ""
    _emit_phase_progress()

    # Unload MV completely
    _unload_mv_verified(monitor)

    # ── Phase 4: Summary (all data already stored per sub-batch) ──
    # Ensure queued DB writes are fully drained
    if _global_db_writer:
        try:
            _global_db_writer.flush(timeout=60.0)
        except Exception as e:
            logger.error(f"[DBQ] flush failed: {e}")

    # WAL checkpoint to flush all writes to main DB
    if _global_sqlite_db:
        _global_sqlite_db.checkpoint()

    # Emit [OK]/[PARTIAL]/[WARN] per file for frontend processedCount tracking
    for pf in parsed:
        if pf.error is None:
            if pf.stored_vv and pf.stored_mv:
                logger.info(f"  [OK] {pf.file_path.name}")
            elif pf.stored_vv or pf.stored_mv:
                logger.info(f"  [OK] {pf.file_path.name} (vv={'Y' if pf.stored_vv else 'N'} mv={'Y' if pf.stored_mv else 'N'})")
            else:
                logger.info(f"  [OK] {pf.file_path.name}")
    _emit_phase_progress()

    parse_errors = sum(1 for pf in parsed if pf.error is not None)

    elapsed = time.perf_counter() - t_total
    monitor.log_status("pipeline_done")
    logger.info(
        f"[DONE] {valid_count} processed, {parse_errors} parse-errors "
        f"(total: {total}, {elapsed:.1f}s)"
    )

    return valid_count, parse_errors, 0


def discover_files(root_dir: Path) -> List[Tuple[Path, str, int, List[str]]]:
    """
    DFS recursive discovery of supported image files.

    Args:
        root_dir: Root directory to scan

    Returns:
        List of (file_path, relative_folder, depth, folder_tags) tuples
    """
    discovered = []
    root_dir = Path(unicodedata.normalize('NFC', str(root_dir.resolve())))

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
                # NFC normalize: macOS returns NFD filenames, unify to NFC for DB
                nfc_entry = Path(unicodedata.normalize('NFC', str(entry)))
                rel = nfc_entry.parent.relative_to(root_dir)
                folder_str = str(rel).replace('\\', '/') if str(rel) != '.' else ''
                folder_tags = [p for p in folder_str.split('/') if p] if folder_str else []
                discovered.append((nfc_entry, folder_str, depth, folder_tags))

    _dfs(root_dir, 0)
    return discovered


def should_skip_file(file_path: Path, db) -> bool:
    """
    Skip only if file is unchanged AND all phases are complete (MC + VV + MV + Structure).
    Partially-processed files (e.g. VV done but MC/MV missing) are NOT skipped.
    """
    info = db.get_file_phase_info(_nfc(file_path.resolve()))
    if info is None:
        return False  # Not in DB

    # mtime check
    stored_mtime = info.get("modified_at")
    if not stored_mtime:
        return False
    current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    if stored_mtime.replace('T', ' ') != current_mtime.replace('T', ' '):
        return False  # File changed

    stat = file_path.stat()
    stored_size = info.get("file_size")
    if stored_size is not None and int(stored_size) != int(stat.st_size):
        return False

    # NOTE:
    # Re-hashing every file during discovery can become the dominant I/O bottleneck.
    # Default to mtime+size skip checks; hash verification is optional.
    from backend.utils.config import get_config as _get_cfg
    verify_hash = bool(_get_cfg().get("batch_processing.skip_verify_hash", False))
    stored_hash = info.get("content_hash")
    if verify_hash and stored_hash:
        try:
            from backend.utils.content_hash import compute_content_hash
            if stored_hash != compute_content_hash(file_path):
                return False
        except Exception:
            return False

    # Missing relative_path means legacy metadata quality gap; do not skip.
    if not info.get("has_relative_path", False):
        return False

    # All phases must be complete to skip
    has_structure = info.get("has_structure", False)
    if not (info["has_mc"] and info["has_vv"] and info["has_mv"] and has_structure):
        return False  # Partially processed — need to continue

    return True


def should_skip_file_enhanced(file_path: Path, db, current_tier: str) -> bool:
    """
    Enhanced skip: mtime + tier + all-phases-complete check.
    Returns True only if file is unchanged, same tier, AND all phases done.
    """
    info = db.get_file_phase_info(_nfc(file_path.resolve()))
    if info is None:
        return False  # Not in DB → must process

    # mtime comparison
    stored_mtime = info.get("modified_at")
    if not stored_mtime:
        return False
    current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    if stored_mtime.replace('T', ' ') != current_mtime.replace('T', ' '):
        return False  # File changed → must reprocess

    stat = file_path.stat()
    stored_size = info.get("file_size")
    if stored_size is not None and int(stored_size) != int(stat.st_size):
        return False

    from backend.utils.config import get_config as _get_cfg
    verify_hash = bool(_get_cfg().get("batch_processing.skip_verify_hash", False))
    stored_hash = info.get("content_hash")
    if verify_hash and stored_hash:
        try:
            from backend.utils.content_hash import compute_content_hash
            if stored_hash != compute_content_hash(file_path):
                return False
        except Exception:
            return False

    # Tier comparison
    stored_tier = info.get("mode_tier")
    if stored_tier and stored_tier != current_tier:
        return False  # Tier changed → must reprocess

    # Missing relative_path means legacy metadata quality gap; do not skip.
    if not info.get("has_relative_path", False):
        return False

    # All phases must be complete to skip
    # v3.6.0: VV phase now includes Structure (DINOv2)
    has_structure = info.get("has_structure", False)
    if not (info["has_mc"] and info["has_vv"] and info["has_mv"] and has_structure):
        return False  # Partially processed

    return True


def _ensure_fts_rebuild_if_needed(db):
    """Best-effort FTS version reconciliation before skip filtering."""
    try:
        bs = db.get_build_status()
        reasons = bs.get("reasons", []) if isinstance(bs, dict) else []
        if any("FTS index version is outdated" in r for r in reasons):
            logger.info("[REBUILD] Detected outdated FTS index; rebuilding before skip filtering...")
            db._rebuild_fts()
    except Exception as e:
        logger.warning(f"[REBUILD] Build-status check failed (non-fatal): {e}")


def _log_rebuild_status(db=None):
    """Log whether rebuild is still required after a run."""
    try:
        status_db = db
        created_status_db = False
        if status_db is None:
            from backend.db.sqlite_client import SQLiteDB
            status_db = SQLiteDB()
            created_status_db = True

        status = status_db.get_build_status()
        if status.get("needs_rebuild"):
            reasons = status.get("reasons", [])
            logger.warning(
                f"[REBUILD][DB] Still required (database-wide): "
                f"{'; '.join(reasons) if reasons else 'unknown reason'}"
            )
        else:
            logger.info("[REBUILD][DB] No rebuild required")

        if created_status_db:
            status_db.close()
    except Exception:
        pass


def _run_common_batch(
    file_infos: List[Tuple[Path, Optional[str], int, List[str]]],
    total: int,
    skip_processed: bool,
    current_tier: Optional[str] = None,
    skip_db=None,
    log_each_skip: bool = False,
):
    """
    Unified executor for both --discover and --files code paths.
    Ensures identical skip criteria and completion logging behavior.
    """
    to_process = []
    skipped = 0

    for fp, folder, depth, tags in file_infos:
        if skip_processed and skip_db is not None:
            if current_tier:
                should_skip = should_skip_file_enhanced(fp, skip_db, current_tier)
            else:
                should_skip = should_skip_file(fp, skip_db)
            if should_skip:
                skipped += 1
                if log_each_skip:
                    tier_txt = f", tier={current_tier}" if current_tier else ""
                    logger.info(f"[SKIP] {fp.name} (unchanged{tier_txt})")
                continue
        to_process.append((fp, folder, depth, tags))

    if skipped > 0:
        logger.info(f"[SKIP] {skipped}/{total} files skipped (unchanged)")

    if len(to_process) == 0:
        logger.info(f"[DONE] 0 processed, 0 errors (total: {total}, {skipped} skipped)")
        _log_rebuild_status(skip_db)
        return

    # Always use phase-based batch path (even for 1 file) so per-phase skip
    # decisions (MC/VV/MV selective reuse) behave consistently.
    stored, parse_errors, store_errors = process_batch_phased(
        to_process,
        allow_phase_skip=skip_processed,
    )
    if len(to_process) == 1:
        logger.info(f"[DONE] {stored} processed (total: {total}, {skipped} skipped)")
    else:
        logger.info(
            f"[DONE] Batch complete: {stored} processed, "
            f"{parse_errors + store_errors} errors (total: {total}, {skipped} skipped)"
        )
    _log_rebuild_status(skip_db)


def run_discovery(
    root_dir: str,
    skip_processed: bool = True,
    batch_size: int = 5,
    current_tier: Optional[str] = None,
):
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

    # Unified skip/filter path
    db = None
    if skip_processed:
        try:
            from backend.db.sqlite_client import SQLiteDB
            db = SQLiteDB()
            _ensure_fts_rebuild_if_needed(db)
        except Exception as e:
            logger.warning(f"Cannot open DB for smart skip: {e}")
    try:
        _run_common_batch(
            discovered,
            total=total,
            skip_processed=skip_processed,
            current_tier=current_tier,
            skip_db=db,
            log_each_skip=False,
        )
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


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
    # Start parent watchdog — auto-exit when Electron (parent) dies unexpectedly.
    # Pipeline/Discover are spawned as detached processes; without this watchdog
    # they would continue running indefinitely after a parent crash.
    try:
        from backend.utils.parent_watchdog import start_parent_watchdog
        start_parent_watchdog()
    except Exception:
        pass

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

    try:
        if args.discover:
            # v3.2: Always use Phase-based pipeline (ignore legacy batch_size)
            run_discovery(args.discover, skip_processed=not args.no_skip, current_tier=tier_name)
        elif args.files:
            # v3.4+: Unified processing path (same core logic as --discover)
            import json
            try:
                file_list = json.loads(args.files)
                total = len(file_list)

                if total == 0:
                    logger.info("[DONE] No files to process")
                else:
                    file_infos = [(Path(fp_str), None, 0, []) for fp_str in file_list]
                    skip_db = SQLiteDB() if not args.no_skip else None
                    try:
                        if skip_db is not None:
                            _ensure_fts_rebuild_if_needed(skip_db)
                        _run_common_batch(
                            file_infos,
                            total=total,
                            skip_processed=not args.no_skip,
                            current_tier=tier_name,
                            skip_db=skip_db,
                            log_each_skip=True,
                        )
                    finally:
                        if skip_db is not None:
                            skip_db.close()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
        elif args.file:
            process_file(Path(args.file))
        elif args.watch:
            start_watcher(args.watch)
        else:
            parser.print_help()
    finally:
        _shutdown_db_writer()

if __name__ == "__main__":
    main()
