"""
Batch Thumbnail Generator with Parallel Processing.
- Memory processing: 5 concurrent (ThreadPool)
- Disk saving: Sequential (Lock)

v3.1: Supports tier-aware preprocessing with aspect ratio preservation and letterbox padding.

Usage:
    python thumbnail_generator.py <file_path>           # Single file (returns base64)
    python thumbnail_generator.py --batch <json_paths>  # Batch mode (saves to disk, returns status)
"""

import sys
import base64
import io

# Force UTF-8 for stdout/stderr to handle generic unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import yaml

# Load config
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
try:
    with open(CONFIG_PATH, encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except:
    CONFIG = {}

DEFAULT_SIZE = CONFIG.get('thumbnail', {}).get('max_size', 1024)  # v3.1: increased from 512
MAX_WORKERS = 5
THUMB_DIR = Path(__file__).parent.parent.parent / "output" / "thumbnails"

# Lock for sequential disk writes
disk_lock = threading.Lock()


def get_thumb_path(file_path: str) -> Path:
    """Return thumbnail path matching parser convention: {stem}_thumb.png"""
    return THUMB_DIR / f"{Path(file_path).stem}_thumb.png"


def load_from_cache(thumb_path: Path) -> str | None:
    if thumb_path.exists():
        with open(thumb_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None


def process_single(file_path: str, size: int = DEFAULT_SIZE) -> tuple[Image.Image | None, str]:
    """Process a single file in memory (no disk write yet)."""
    path = Path(file_path)
    ext = path.suffix.lower()

    # Check existing thumbnail first (shared with parsers)
    thumb_path = get_thumb_path(file_path)
    cached = load_from_cache(thumb_path)
    if cached:
        return None, cached  # Already exists, return base64
    
    try:
        if ext == '.psd':
            from psd_tools import PSDImage
            psd = PSDImage.open(file_path)
            img = psd.composite()
        else:
            img = Image.open(file_path)
        
        # v3.1: Keep RGBA for disk storage (transparency preservation)
        # Composite to RGB will be done in-memory during indexing
        if img.mode == 'P':
            img = img.convert('RGBA')
        elif img.mode not in ('RGB', 'RGBA'):
            # Convert grayscale/other modes to RGB
            img = img.convert('RGB')
        
        # Resize
        w, h = img.size
        if w > h:
            new_w, new_h = size, int(h * (size / w))
        else:
            new_h, new_w = size, int(w * (size / h))
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img, file_path
        
    except Exception as e:
        print(f"ERROR processing {file_path}: {e}", file=sys.stderr)
        return None, file_path


def save_to_cache_sequential(img: Image.Image, file_path: str, size: int) -> str:
    """Save thumbnail to output/thumbnails/ (sequential via lock) and return base64."""
    thumb_path = get_thumb_path(file_path)

    with disk_lock:
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb_path, format='PNG', optimize=True)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def generate_single(file_path: str, size: int = DEFAULT_SIZE) -> str:
    """Generate single thumbnail (for single file mode)."""
    img, result = process_single(file_path, size)
    
    if img is None:
        return result if result != file_path else ""
    
    return save_to_cache_sequential(img, file_path, size)


def generate_batch(file_paths: list[str], size: int = DEFAULT_SIZE) -> dict:
    """Generate thumbnails in parallel, save sequentially."""
    results = {}
    pending_saves = []
    
    # Phase 1: Parallel processing in memory
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, fp, size): fp for fp in file_paths}
        
        for future in as_completed(futures):
            fp = futures[future]
            try:
                img, result = future.result()
                if img is None:
                    # Already cached or error
                    results[fp] = result if result != fp else None
                else:
                    pending_saves.append((img, fp))
            except Exception as e:
                print(f"ERROR: {fp}: {e}", file=sys.stderr)
                results[fp] = None
    
    # Phase 2: Sequential disk saves
    for img, fp in pending_saves:
        try:
            b64 = save_to_cache_sequential(img, fp, size)
            results[fp] = b64
        except Exception as e:
            print(f"ERROR saving {fp}: {e}", file=sys.stderr)
            results[fp] = None
    
    return results


# ══════════════════════════════════════════════════════════════════
# v3.1: Tier-Aware Preprocessing with Aspect Ratio Preservation
# ══════════════════════════════════════════════════════════════════

def _parse_color(hex_color: str) -> tuple:
    """Convert hex color to RGB/RGBA tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    elif len(hex_color) == 8:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    else:
        return (255, 255, 255)  # White fallback


def generate_thumbnail_with_tier(file_path: str) -> Image.Image:
    """
    Tier-aware thumbnail generation with aspect ratio preservation.

    v3.1: Applies tier-specific preprocessing:
    - Standard: max_edge=512, letterbox padding
    - Pro: max_edge=768, letterbox padding
    - Ultra: max_edge=1024, letterbox padding

    Args:
        file_path: Path to image file (PSD/PNG/JPG)

    Returns:
        PIL Image (RGBA or RGB) with aspect ratio preserved and letterbox padding

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If image loading/processing fails
    """
    from backend.utils.tier_config import get_active_tier

    tier_name, tier_config = get_active_tier()
    preprocess = tier_config.get("preprocess", {})

    max_edge = preprocess.get("max_edge", 768)
    mode = preprocess.get("aspect_ratio_mode", "contain")
    padding_color = preprocess.get("padding_color", "#FFFFFF")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load original image
    ext = path.suffix.lower()
    if ext == '.psd':
        from psd_tools import PSDImage
        psd = PSDImage.open(file_path)
        img = psd.composite()
    else:
        img = Image.open(file_path)

    # Convert palette/grayscale to RGBA/RGB
    if img.mode == 'P':
        img = img.convert('RGBA')
    elif img.mode not in ('RGB', 'RGBA', 'LA'):
        img = img.convert('RGB')

    # Aspect Ratio preservation with contain mode
    w, h = img.size
    scale = min(max_edge / w, max_edge / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if mode == "contain":
        # Create square canvas with letterbox padding
        canvas_mode = "RGBA" if img.mode in ("RGBA", "LA") else "RGB"
        canvas = Image.new(
            canvas_mode,
            (max_edge, max_edge),
            _parse_color(padding_color)
        )

        # Center the resized image
        offset_x = (max_edge - new_w) // 2
        offset_y = (max_edge - new_h) // 2
        canvas.paste(img_resized, (offset_x, offset_y))

        return canvas
    else:
        # Direct resize without padding (legacy mode)
        return img_resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?', help='Single file path')
    parser.add_argument('--batch', type=str, help='JSON array of file paths for batch mode')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE)
    args = parser.parse_args()

    if args.batch:
        # Batch mode
        paths = json.loads(args.batch)
        results = generate_batch(paths, args.size)
        print(json.dumps(results))
    elif args.file_path:
        # Single file mode
        result = generate_single(args.file_path, args.size)
        if result:
            print(result)
            sys.exit(0)
        sys.exit(1)
    else:
        print("Usage: thumbnail_generator.py <file> or --batch '[...]'", file=sys.stderr)
        sys.exit(1)
