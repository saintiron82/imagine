"""
Env Installer - Handles dependency installation, model download, and SQLite setup.

This script ensures all dependencies are installed including:
- Python packages (torch, transformers, sqlite-vec, etc.)
- SigLIP2 VV model (tier-based)
- Ollama VLM + MV models
- SQLite + sqlite-vec setup verification
"""
import sys
import subprocess
import json
import logging
from pathlib import Path
import platform

# Add project root to sys.path for module resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("installer")

REQUIRED_PACKAGES = [
    "torch",
    "sqlite-vec",          # SQLite vector extension
    "transformers",        # SigLIP2, Qwen3-VL
    "accelerate",          # Model loading
    "numpy",               # Array operations
    "deep-translator",
    "pillow",
    "requests",
    "pydantic",
    "watchdog",
    "tqdm",
    "psd-tools",
    "exifread",
    "PyYAML",
]

def check_imports():
    """Check if packages are importable."""
    status = {}
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            # Handle package name vs import name differences
            import_name = pkg.replace("-", "_")
            if pkg == "deep-translator": import_name = "deep_translator"
            if pkg == "pillow": import_name = "PIL"
            if pkg == "PyYAML": import_name = "yaml"

            __import__(import_name)
            status[pkg] = True
        except ImportError:
            status[pkg] = False
            all_ok = False
    return status, all_ok

def _get_tier_config():
    """Get active tier name and config from config.yaml."""
    try:
        from backend.utils.tier_config import get_active_tier
        return get_active_tier()
    except Exception:
        return "standard", {}

def check_model():
    """Check if SigLIP2 VV model is cached (tier-based)."""
    try:
        tier_name, tier_config = _get_tier_config()
        model_name = tier_config.get("visual", {}).get("model", "google/siglip2-base-patch16-224")

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # HuggingFace caches models as "models--org--name"
        safe_name = model_name.replace("/", "--")
        model_dirs = list(cache_dir.glob(f"models--{safe_name}"))
        cached = len(model_dirs) > 0
        return cached, model_name
    except Exception:
        return False, "unknown"

def check_sqlite():
    """Check if SQLite version supports sqlite-vec (3.41+)."""
    try:
        import sqlite3
        version_tuple = sqlite3.sqlite_version_info
        if version_tuple >= (3, 41, 0):
            return True, f"SQLite {sqlite3.sqlite_version}"
        else:
            return False, f"SQLite {sqlite3.sqlite_version} (requires 3.41+)"
    except Exception as e:
        return False, f"SQLite check failed: {str(e)}"

def check_sqlitevec():
    """Check if sqlite-vec extension is available."""
    try:
        import sqlite3
        import sqlite_vec
        # Try loading extension
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        cursor = conn.execute("SELECT vec_version()")
        version = cursor.fetchone()[0]
        conn.close()
        return True, f"sqlite-vec {version}"
    except ImportError:
        return False, "sqlite-vec not installed"
    except Exception as e:
        return False, f"Extension load failed: {str(e)}"

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []

def check_ollama_models(tier_name=None):
    """Check if required Ollama models are available for the given tier."""
    tier_models = {
        "standard": ["qwen3-vl:2b", "qwen3-embedding:0.6b"],
        "pro": ["qwen3-embedding:0.6b"],  # pro VLM uses transformers
        "ultra": ["qwen3-vl:8b", "qwen3-embedding:8b"],
    }

    if tier_name is None:
        tier_name, _ = _get_tier_config()

    required = tier_models.get(tier_name, [])
    running, installed = check_ollama()

    if not running:
        return False, required, []

    # Normalize model names for comparison (strip :latest etc.)
    installed_base = set()
    for m in installed:
        installed_base.add(m)
        if ":" in m:
            installed_base.add(m.split(":")[0])

    missing = []
    for req in required:
        req_base = req.split(":")[0] if ":" in req else req
        found = any(req in installed_base or req_base in inst for inst in installed_base)
        if not found:
            missing.append(req)

    return len(missing) == 0, required, missing

def pull_ollama_models(tier_name=None):
    """Pull required Ollama models for the given tier."""
    ok, required, missing = check_ollama_models(tier_name)
    if ok:
        logger.info(f"✅ All Ollama models ready: {', '.join(required)}")
        return True

    running, _ = check_ollama()
    if not running:
        logger.error("❌ Ollama is not running. Please start Ollama first.")
        logger.error("   Download: https://ollama.com/download")
        return False

    for model in missing:
        logger.info(f"Pulling Ollama model: {model} ...")
        try:
            subprocess.check_call(["ollama", "pull", model], timeout=600)
            logger.info(f"  ✅ {model} downloaded")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"  ❌ Failed to pull {model}: {e}")
            return False

    return True

def install_packages():
    """Install packages via pip."""
    logger.info("Installing dependencies...")
    try:
        # Detect if running in virtualenv
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        # Build pip command
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        # Only use --user if NOT in virtualenv
        if not in_venv:
            cmd.append("--user")
        cmd.extend(REQUIRED_PACKAGES)

        subprocess.check_call(cmd)
        logger.info("✅ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {e}")
        return False

def download_model():
    """Download SigLIP2 VV model (tier-based)."""
    tier_name, tier_config = _get_tier_config()
    model_name = tier_config.get("visual", {}).get("model", "google/siglip2-base-patch16-224")

    logger.info(f"Downloading VV model ({tier_name} tier): {model_name}")
    logger.info("This may take a while on first run...")
    try:
        from transformers import AutoModel, AutoProcessor
        AutoProcessor.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        logger.info(f"✅ Model downloaded and verified: {model_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        return False

def setup_sqlite():
    """Guide user on SQLite setup."""
    logger.info("\n" + "="*80)
    logger.info("SQLite Setup")
    logger.info("="*80)

    sqlite_ok, sqlite_msg = check_sqlite()
    sqlitevec_ok, sqlitevec_msg = check_sqlitevec()

    if not sqlite_ok:
        logger.info(f"\n❌ {sqlite_msg}")
        logger.info("\nSQLite 3.41+ is required for vector support.")
        logger.info("\nOptions:")
        logger.info("  - Windows: SQLite is built into Python, update Python to 3.11+")
        logger.info("  - Linux: Update system SQLite package")
        logger.info("  - Build from source: https://www.sqlite.org/download.html")
    else:
        logger.info(f"\n✅ {sqlite_msg}")

    if not sqlitevec_ok:
        logger.info(f"\n❌ {sqlitevec_msg}")
        logger.info("\nTo install sqlite-vec:")
        logger.info("  pip install sqlite-vec")
    else:
        logger.info(f"✅ {sqlitevec_msg}")

    logger.info("="*80)

def init_database():
    """Initialize SQLite database schema."""
    logger.info("\n" + "="*80)
    logger.info("Initializing SQLite Database")
    logger.info("="*80)

    try:
        from backend.db.sqlite_client import SQLiteDB

        logger.info("Creating database connection...")
        db = SQLiteDB()

        logger.info("Creating schema, tables, and indexes...")
        db.init_schema()

        logger.info("Verifying sqlite-vec extension...")
        try:
            cursor = db.conn.execute("SELECT vec_version()")
            vec_ver = cursor.fetchone()[0]
            logger.info(f"  ✅ sqlite-vec version: {vec_ver}")
        except:
            logger.warning("  ⚠️ vec_version() not available - sqlite-vec may not be loaded")
            logger.warning("     Vector search will not work!")

        # Check database stats
        stats = db.get_stats()
        logger.info(f"\nDatabase Statistics:")
        logger.info(f"  Total files:  {stats['total_files']}")
        logger.info(f"  Total layers: {stats['total_layers']}")

        db.close()
        logger.info("\n✅ Database initialized successfully!")
        logger.info(f"   Location: {db.db_path}")
        logger.info("="*80)
        return True

    except Exception as e:
        logger.error(f"\n❌ Database initialization failed: {e}")
        logger.error("\nPlease ensure:")
        logger.error("  - SQLite version 3.41+ is available")
        logger.error("  - sqlite-vec is installed: pip install sqlite-vec")
        logger.info("="*80)
        return False

def check_gpu():
    """Check GPU/VRAM status (CUDA, Apple Silicon MPS, or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            return True, f"{name} ({vram} MB VRAM)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "Apple Silicon MPS (GPU accelerated)"
        return False, "No GPU detected (CPU mode)"
    except ImportError:
        return False, "PyTorch not installed"

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ImageParser Installer - Setup dependencies and database"
    )
    parser.add_argument("--check", action="store_true",
                       help="Check installation status and exit with JSON")
    parser.add_argument("--install", action="store_true",
                       help="Install Python dependencies")
    parser.add_argument("--download-model", action="store_true",
                       help="Download SigLIP2 VV model")
    parser.add_argument("--setup-sqlite", action="store_true",
                       help="Show SQLite setup guide")
    parser.add_argument("--init-db", action="store_true",
                       help="Initialize SQLite database schema")
    parser.add_argument("--setup-ollama", action="store_true",
                       help="Check and pull required Ollama models")
    parser.add_argument("--full-setup", action="store_true",
                       help="Run full setup (install + models + db)")
    args = parser.parse_args()

    if args.check:
        deps_status, deps_ok = check_imports()
        model_ok, model_name = check_model()
        sqlite_ok, sqlite_msg = check_sqlite()
        sqlitevec_ok, sqlitevec_msg = check_sqlitevec()
        ollama_ok, ollama_models = check_ollama()
        ollama_models_ok, required_models, missing_models = check_ollama_models()
        gpu_ok, gpu_msg = check_gpu()
        tier_name, _ = _get_tier_config()

        result = {
            "dependencies": deps_status,
            "dependencies_ok": deps_ok,
            "visual_model": model_name,
            "visual_model_cached": model_ok,
            "sqlite_ok": sqlite_ok,
            "sqlite_message": sqlite_msg,
            "sqlitevec_ok": sqlitevec_ok,
            "sqlitevec_message": sqlitevec_msg,
            "ollama_running": ollama_ok,
            "ollama_models_ok": ollama_models_ok,
            "ollama_required": required_models,
            "ollama_missing": missing_models,
            "gpu": gpu_msg,
            "gpu_available": gpu_ok,
            "tier": tier_name,
            "platform": platform.system(),
        }

        print(json.dumps(result, indent=2))

        # Print human-readable summary
        logger.info("\n" + "="*80)
        logger.info(f"Installation Status (tier: {tier_name})")
        logger.info("="*80)
        logger.info(f"Python Dependencies:  {'✅ OK' if deps_ok else '❌ Missing'}")
        logger.info(f"Visual Model Cached:  {'✅ ' + model_name if model_ok else '❌ ' + model_name}")
        logger.info(f"SQLite:               {'✅ ' + sqlite_msg if sqlite_ok else '❌ ' + sqlite_msg}")
        logger.info(f"sqlite-vec Extension: {'✅ ' + sqlitevec_msg if sqlitevec_ok else '❌ ' + sqlitevec_msg}")
        logger.info(f"Ollama Running:       {'✅ Yes' if ollama_ok else '❌ No'}")
        logger.info(f"Ollama Models:        {'✅ Ready' if ollama_models_ok else '❌ Missing: ' + ', '.join(missing_models)}")
        logger.info(f"GPU:                  {gpu_msg}")
        logger.info("="*80)

        if not deps_ok:
            logger.info("\nTo install dependencies: python backend/setup/installer.py --install")
        if not model_ok:
            logger.info("To download model:       python backend/setup/installer.py --download-model")
        if not ollama_ok:
            logger.info("Ollama not running:      Start Ollama (https://ollama.com/download)")
        elif not ollama_models_ok:
            logger.info("To pull Ollama models:   python backend/setup/installer.py --setup-ollama")
        if not sqlite_ok or not sqlitevec_ok:
            logger.info("To setup SQLite:         python backend/setup/installer.py --setup-sqlite")
        if sqlite_ok and sqlitevec_ok:
            logger.info("To initialize database:  python backend/setup/installer.py --init-db")

        return

    if args.full_setup:
        tier_name, _ = _get_tier_config()
        logger.info(f"Running full setup (tier: {tier_name})...")

        # 1. Install packages
        if not install_packages():
            logger.error("❌ Package installation failed!")
            sys.exit(1)

        # 2. Check SQLite
        sqlite_ok, sqlite_msg = check_sqlite()
        sqlitevec_ok, sqlitevec_msg = check_sqlitevec()

        if not sqlite_ok or not sqlitevec_ok:
            logger.warning("⚠️ SQLite setup incomplete")
            setup_sqlite()
            logger.info("\nAfter fixing SQLite setup, run:")
            logger.info("  python backend/setup/installer.py --init-db")
        else:
            # Initialize database
            if not init_database():
                sys.exit(1)

        # 3. Download SigLIP2 model
        if not download_model():
            logger.warning("⚠️ Visual model download failed - will retry on first use")

        # 4. Setup Ollama models
        ollama_ok, _ = check_ollama()
        if ollama_ok:
            pull_ollama_models(tier_name)
        else:
            logger.warning("⚠️ Ollama is not running - skipping model pull")
            logger.warning("   Install Ollama: https://ollama.com/download")
            logger.warning("   Then run: python backend/setup/installer.py --setup-ollama")

        logger.info("\n✅ Full setup complete!")
        return

    if args.install:
        if not install_packages():
            sys.exit(1)

    if args.download_model:
        if not download_model():
            sys.exit(1)

    if args.setup_sqlite:
        setup_sqlite()

    if args.init_db:
        if not init_database():
            sys.exit(1)

    if args.setup_ollama:
        if not pull_ollama_models():
            sys.exit(1)

if __name__ == "__main__":
    main()
