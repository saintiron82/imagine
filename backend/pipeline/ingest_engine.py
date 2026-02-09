"""
Ingest Engine - Main Entry Point for Data Pipeline.

Handles:
1. File detection (CLI or Watchdog)
2. Parser selection (Factory Pattern)
3. Execution and Result Logging
"""

import argparse
import logging
import sys
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
# Translation removed from ingest pipeline.
# Cross-language search is handled at query time by QueryDecomposer.

# Force UTF-8 for stdout/stderr to handle generic unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
from backend.utils.auto_batch_calibrator import AutoBatchCalibrator


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


class ParserFactory:
    """Factory to select the correct parser for a file."""
    
    _parsers = [PSDParser, ImageParser]
    
    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[BaseParser]:
        """Return an instantiated parser capable of handling the file."""
        for parser_cls in cls._parsers:
            if parser_cls.can_parse(file_path):
                return parser_cls(output_dir=Path("output"))
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
            # --file mode: extract folder metadata from file path
            # v3.1 Context Injection: ensure folder_path is always populated
            parent = file_path.parent
            if parent.name and parent.name not in (".", ""):
                meta.folder_path = str(parent)
                # Calculate depth (number of parent directories)
                meta.folder_depth = len(parent.parts) - 1 if len(parent.parts) > 1 else 0
                # Use folder name as tag
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

                # v3 P0: path normalization (POSIX)
                normalized = str(file_path).replace('\\', '/')
                # Derive storage_root + relative_path
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

                # v3 P0: embedding version
                from backend.utils.config import get_config
                from backend.utils.tier_config import get_active_tier

                cfg = get_config()
                tier_name, tier_config = get_active_tier()

                # v3.1: Tier metadata
                meta.mode_tier = tier_name
                meta.caption_model = tier_config.get("vlm", {}).get("model", "")
                meta.text_embed_model = tier_config.get("text_embed", {}).get("model", "")
                meta.runtime_version = cfg.get("runtime.ollama_version", "")
                meta.preprocess_params = tier_config.get("preprocess", {})

                # Visual encoder model (tier-aware)
                meta.embedding_model = tier_config.get("visual", {}).get("model") or cfg.get(
                    "embedding.visual.model", "google/siglip2-so400m-patch14-384"
                )
                meta.embedding_version = 1

                # Save metadata with AI fields
                parser._save_json(meta, file_path)

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

        # === STEP 3/4: Embedding ===
        step3_start = time.time()
        logger.info(f"STEP 3/4 Embedding")
        # === 5. PostgreSQL Storage (Vector + Metadata) ===
        try:
            # Lazy load SQLite client and CLIP model
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

            # T-axis: Generate text embedding from caption + tags
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
                            logger.info(f"   T-axis: text embedding stored ({len(text_emb)}-dim)")
            except Exception as e:
                logger.warning(f"T-axis embedding failed (non-fatal): {e}")

        except Exception as e:
            logger.error(f"SQLite Storage Failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        step4_duration = time.time() - step4_start
        logger.info(f"STEP 4/4 completed in {step4_duration:.2f}s")

        total_duration = time.time() - step_start
        logger.info(f"[OK] Total processing time: {total_duration:.2f}s")
        logger.info(f"[OK] Parsed successfully in {parse_duration:.2f}s")
        logger.info(f"   Format: {meta.format}, Layers: {meta.layer_count}")
        logger.info(f"   Tags: {meta.semantic_tags}")
        if meta.thumbnail_url:
            logger.info(f"   Thumb: {meta.thumbnail_url}")
    else:
        logger.error(f"[FAIL] Parsing failed: {result.errors}")


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


def run_discovery(root_dir: str, skip_processed: bool = True, batch_size: int = 5):
    """
    DFS discover all supported files and process them.

    Args:
        root_dir: Root directory to scan
        skip_processed: Skip files unchanged since last processing
        batch_size: Number of files to process concurrently (default: 5)
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Directory not found: {root_dir}")
        return

    # Phase 1: DFS discovery
    logger.info(f"[DISCOVER] Scanning: {root_path}")
    discovered = discover_files(root_path)
    total = len(discovered)
    logger.info(f"[DISCOVER] Found {total} supported files")

    if total == 0:
        return

    # Phase 2: Filter files (smart skip)
    db = None
    if skip_processed:
        try:
            from backend.db.sqlite_client import SQLiteDB
            db = SQLiteDB()
        except Exception as e:
            logger.warning(f"Cannot open DB for smart skip: {e}")

    # Filter out skipped files
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

    # Phase 3: Batch processing with ThreadPoolExecutor
    def process_wrapper(args):
        """Wrapper for parallel processing"""
        idx, fp, folder, depth, tags = args
        logger.info(f"[{idx}/{len(to_process)}] Processing: {fp.name}")
        try:
            process_file(fp, folder_path=folder, folder_depth=depth, folder_tags=tags)
            return ('success', fp.name)
        except Exception as e:
            logger.error(f"[ERROR] {fp.name}: {e}")
            return ('error', fp.name, str(e))

    processed, errors = 0, 0

    if batch_size > 1 and len(to_process) > 1:
        logger.info(f"[BATCH] Processing {len(to_process)} files with batch_size={batch_size}")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks
            futures = []
            for idx, (fp, folder, depth, tags) in enumerate(to_process, 1):
                future = executor.submit(process_wrapper, (idx, fp, folder, depth, tags))
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result[0] == 'success':
                    processed += 1
                elif result[0] == 'error':
                    errors += 1
    else:
        # Sequential processing for batch_size=1
        for idx, (fp, folder, depth, tags) in enumerate(to_process, 1):
            result = process_wrapper((idx, fp, folder, depth, tags))
            if result[0] == 'success':
                processed += 1
            elif result[0] == 'error':
                errors += 1

    logger.info(f"[DONE] {processed} processed, {skipped} skipped, {errors} errors (total: {total})")


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
    parser.add_argument("--batch-size", type=parse_batch_size, default='adaptive',
                       help="Batch size: integer (e.g. 5), 'auto' (use cached optimal), or 'adaptive' (find optimal at runtime, default)")

    args = parser.parse_args()

    # Check batch_processing.enabled config
    from backend.utils.config import get_config
    config = get_config()
    batch_enabled = config.get("batch_processing", {}).get("enabled", True)

    # ===== TIER COMPATIBILITY CHECK =====
    from backend.utils.tier_config import get_active_tier
    from backend.db.sqlite_client import SQLiteDB

    tier_name, tier_config = get_active_tier()
    visual_config = tier_config.get("visual", {})
    expected_dimension = visual_config.get("dimensions", 768)

    logger.info(f"[TIER CHECK] Current tier: {tier_name}, Expected dimension: {expected_dimension}")

    # Check DB compatibility
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

        # Show user prompt if available
        if compat['user_prompt']:
            logger.error("Details:")
            for line in compat['user_prompt'].split('\n'):
                logger.error(f"  {line}")
            logger.error("")

        # Show migration steps
        logger.error("Migration Steps:")
        steps = get_migration_steps(compat['db_tier'], compat['current_tier'])
        for step in steps:
            logger.error(f"  {step}")
        logger.error("")

        # Alternative: revert config
        logger.error("OR revert config.yaml to previous tier:")
        logger.error(f"  ai_mode.override: {compat['db_tier']}")
        logger.error("=" * 60)

        # Block execution
        logger.error("\n⛔ Execution blocked. Please migrate tier or revert config.")
        sys.exit(1)

    logger.info(f"[TIER CHECK] ✅ Compatibility OK")
    logger.info(f"  DB Tier: {compat['db_tier'] or 'empty'}")
    logger.info(f"  Current Tier: {compat['current_tier']}")
    logger.info(f"  Action: {compat['action']}")
    db.close()

    if not batch_enabled and args.batch_size in ['auto', 'adaptive']:
        logger.info("[CONFIG] Batch processing disabled in config.yaml")
        logger.info("[CONFIG] Forcing batch_size=1 (sequential processing)")
        args.batch_size = 1

    # AUTO/ADAPTIVE batch size
    if args.batch_size in ['auto', 'adaptive']:
        from backend.utils.tier_config import get_active_tier
        from backend.utils.platform_detector import get_optimal_batch_size, get_platform_info
        import platform as platform_module

        tier_name, _ = get_active_tier()

        if args.batch_size == 'auto':
            # Use cached optimal batch size
            logger.info("[AUTO] Using cached optimal batch size")

            cached = AutoBatchCalibrator.load_latest_calibration(tier_name)
            if cached:
                logger.info(f"[AUTO] Cached batch size for {tier_name}: {cached['recommended']}")
                args.batch_size = cached['recommended']
            else:
                # Fallback to platform detector
                logger.info("[AUTO] No cache found, using platform detector")
                backend = 'auto'  # Let platform detector decide
                args.batch_size = get_optimal_batch_size(backend, tier_name)
                logger.info(f"[AUTO] Platform-recommended batch size: {args.batch_size}")

        elif args.batch_size == 'adaptive':
            # Adaptive mode: Start with 1, increase gradually
            logger.info("[ADAPTIVE] Adaptive batch sizing enabled")
            logger.info("[ADAPTIVE] Will start with batch_size=1 and increase during processing")

            platform_info = get_platform_info()
            logger.info(f"[ADAPTIVE] Platform: {platform_info['os']}, Tier: {tier_name}")

            # Note: Actual adaptive logic will be applied during processing
            args.batch_size = 'adaptive'  # Keep as string for later processing

    if args.discover:
        if args.batch_size == 'adaptive':
            # Adaptive batch processing
            from backend.utils.adaptive_batch_processor import AdaptiveBatchProcessor
            logger.info("[ADAPTIVE] Using adaptive batch processor")

            # Discover files first
            root_path = Path(args.discover).resolve()
            discovered = discover_files(root_path)
            total = len(discovered)
            logger.info(f"[DISCOVER] Found {total} supported files")

            if total == 0:
                return

            # Filter files (smart skip)
            db = None
            if not args.no_skip:
                try:
                    from backend.db.sqlite_client import SQLiteDB
                    db = SQLiteDB()
                except Exception as e:
                    logger.warning(f"Cannot open DB for smart skip: {e}")

            to_process = []
            skipped = 0
            for fp, folder, depth, tags in discovered:
                if not args.no_skip and db and should_skip_file(fp, db):
                    skipped += 1
                    logger.debug(f"[SKIP] {fp.name} (unchanged)")
                else:
                    to_process.append((fp, folder, depth, tags))

            if skipped > 0:
                logger.info(f"[SKIP] {skipped} files unchanged, {len(to_process)} to process")

            if len(to_process) == 0:
                logger.info("[DONE] No files to process")
                return

            # Adaptive processing
            def process_single(args_tuple):
                fp, folder, depth, tags = args_tuple
                process_file(fp, folder_path=folder, folder_depth=depth, folder_tags=tags)
                return fp

            processor = AdaptiveBatchProcessor(
                process_func=process_single,
                platform=platform_module.system(),
                tier=tier_name
            )

            results, optimal_batch_size = processor.process_adaptive(to_process, use_cache=True)

            logger.info(f"[ADAPTIVE] Optimal batch size found: {optimal_batch_size}")
            logger.info(f"\n{processor.get_metrics_summary()}")
            logger.info(f"[DONE] {len(to_process)} processed, {skipped} skipped (total: {total})")
        else:
            # Standard batch processing
            run_discovery(args.discover, skip_processed=not args.no_skip, batch_size=args.batch_size)
    elif args.files:
        # Batch mode - process files with parallel processing
        import json
        try:
            file_list = json.loads(args.files)
            total = len(file_list)
            logger.info(f"[BATCH] Processing {total} files with batch_size={args.batch_size}...")

            def process_indexed_file(args_tuple):
                idx, fp = args_tuple
                logger.info(f"[{idx}/{total}] Processing: {fp}")
                try:
                    process_file(Path(fp))
                    return ('success', fp)
                except Exception as e:
                    logger.error(f"[ERROR] {fp}: {e}")
                    return ('error', fp, str(e))

            processed, errors = 0, 0
            if args.batch_size == 'adaptive':
                # Adaptive batch processing
                from backend.utils.adaptive_batch_processor import AdaptiveBatchProcessor
                logger.info("[ADAPTIVE] Using adaptive batch processor")

                def process_single(fp):
                    process_file(Path(fp))
                    return fp

                processor = AdaptiveBatchProcessor(
                    process_func=process_single,
                    platform=platform_module.system(),
                    tier=tier_name
                )

                results, optimal_batch_size = processor.process_adaptive(file_list, use_cache=True)
                processed = len(file_list)
                errors = 0

                logger.info(f"[ADAPTIVE] Optimal batch size found: {optimal_batch_size}")
                logger.info(f"\n{processor.get_metrics_summary()}")
            elif args.batch_size > 1 and total > 1:
                with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
                    futures = [executor.submit(process_indexed_file, (i, fp)) for i, fp in enumerate(file_list, 1)]
                    for future in as_completed(futures):
                        result = future.result()
                        if result[0] == 'success':
                            processed += 1
                        else:
                            errors += 1
            else:
                for i, fp in enumerate(file_list, 1):
                    result = process_indexed_file((i, fp))
                    if result[0] == 'success':
                        processed += 1
                    else:
                        errors += 1

            logger.info(f"[DONE] Batch complete: {processed} processed, {errors} errors (total: {total})")
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
