"""
Env Installer - Handles dependency installation, model download, and SQLite setup.

This script ensures all dependencies are installed including:
- Python packages (torch, sentence-transformers, sqlite-vec, etc.)
- CLIP model (clip-ViT-L-14)
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
    "sentence-transformers",
    "deep-translator",
    "pillow",
    "requests",
    "pydantic",
    "watchdog",
    "tqdm",
    "psd-tools",
    "exifread",
]

def check_imports():
    """Check if packages are importable."""
    status = {}
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            # Handle package name vs import name differences
            import_name = pkg.replace("-", "_")
            if pkg == "sentence-transformers": import_name = "sentence_transformers"
            if pkg == "deep-translator": import_name = "deep_translator"
            if pkg == "pillow": import_name = "PIL"

            __import__(import_name)
            status[pkg] = True
        except ImportError:
            status[pkg] = False
            all_ok = False
    return status, all_ok

def check_model():
    """Check if CLIP model is cached."""
    try:
        from sentence_transformers import SentenceTransformer
        # Check if model files exist in cache
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Look for clip-ViT-L-14 model files
        model_dirs = list(cache_dir.glob("*clip-ViT-L-14*"))
        return len(model_dirs) > 0
    except:
        return False

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
    """Download CLIP model."""
    logger.info("Downloading AI Model (CLIP-ViT-L-14)... This may take a while.")
    try:
        from sentence_transformers import SentenceTransformer
        # This triggers download if not cached
        SentenceTransformer('clip-ViT-L-14')
        logger.info("✅ Model downloaded and verified.")
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
                       help="Download CLIP AI model")
    parser.add_argument("--setup-sqlite", action="store_true",
                       help="Show SQLite setup guide")
    parser.add_argument("--init-db", action="store_true",
                       help="Initialize SQLite database schema")
    parser.add_argument("--full-setup", action="store_true",
                       help="Run full setup (install + download model + init db)")
    args = parser.parse_args()

    if args.check:
        deps_status, deps_ok = check_imports()
        model_ok = check_model()
        sqlite_ok, sqlite_msg = check_sqlite()
        sqlitevec_ok, sqlitevec_msg = check_sqlitevec()

        result = {
            "dependencies": deps_status,
            "dependencies_ok": deps_ok,
            "model_cached": model_ok,
            "sqlite_ok": sqlite_ok,
            "sqlite_message": sqlite_msg,
            "sqlitevec_ok": sqlitevec_ok,
            "sqlitevec_message": sqlitevec_msg,
            "platform": platform.system(),
        }

        print(json.dumps(result, indent=2))

        # Print human-readable summary
        logger.info("\n" + "="*80)
        logger.info("Installation Status")
        logger.info("="*80)
        logger.info(f"Python Dependencies: {'✅ OK' if deps_ok else '❌ Missing'}")
        logger.info(f"CLIP Model Cached:   {'✅ Yes' if model_ok else '❌ No'}")
        logger.info(f"SQLite:              {'✅ ' + sqlite_msg if sqlite_ok else '❌ ' + sqlite_msg}")
        logger.info(f"sqlite-vec Extension:{'✅ ' + sqlitevec_msg if sqlitevec_ok else '❌ ' + sqlitevec_msg}")
        logger.info("="*80)

        if not deps_ok:
            logger.info("\nTo install dependencies: python backend/setup/installer.py --install")
        if not model_ok:
            logger.info("To download model:       python backend/setup/installer.py --download-model")
        if not sqlite_ok or not sqlitevec_ok:
            logger.info("To setup SQLite:         python backend/setup/installer.py --setup-sqlite")
        if sqlite_ok and sqlitevec_ok:
            logger.info("To initialize database:  python backend/setup/installer.py --init-db")

        return

    if args.full_setup:
        logger.info("Running full setup...")

        # Install packages
        if not install_packages():
            logger.error("❌ Package installation failed!")
            sys.exit(1)

        # Download model
        if not download_model():
            logger.error("❌ Model download failed!")
            sys.exit(1)

        # Check SQLite
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

if __name__ == "__main__":
    main()
