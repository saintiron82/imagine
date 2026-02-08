# ImageParser Installation Guide

Complete installation guide for setting up ImageParser on a new machine.

## Prerequisites

- **Python 3.11+** (Python 3.11.9 recommended)
- **Git** (for cloning the repository)
- **Docker Desktop** (recommended) OR **PostgreSQL 16+** (manual installation)

## Quick Start (Recommended)

### 1. Clone Repository

```powershell
git clone <repository-url>
cd ImageParser
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify activation (should show .venv path)
where python
```

### 3. Run Full Setup

```powershell
# This will:
# - Install all Python dependencies
# - Download CLIP AI model (~1.7GB)
# - Guide you through PostgreSQL setup
# - Initialize database schema

python backend/setup/installer.py --full-setup
```

### 4. Configure Environment (Optional)

```powershell
# Copy environment template
copy .env.example .env

# Edit .env file to customize settings
# Key settings:
# - VISION_BACKEND: 'transformers' (default) or 'ollama' (memory-efficient)
# - VISION_MODEL: AI model to use for image analysis
# - POSTGRES_HOST, POSTGRES_PORT, etc: Database connection

notepad .env
```

**Default Configuration** (no .env needed):
- Vision Backend: Transformers with Qwen2-VL-2B-Instruct
- PostgreSQL: localhost:5432 (Docker default)
- Output: output/ directory

**For Deployment** (recommended):
- Set `VISION_BACKEND=ollama` for memory efficiency
- Install Ollama: See [docs/ollama_setup.md](docs/ollama_setup.md)

### 5. Setup PostgreSQL

The installer will guide you, but here are the options:

#### Option A: Docker (Easiest - Recommended)

```powershell
# Install Docker Desktop for Windows
# Download: https://www.docker.com/products/docker-desktop/

# Start PostgreSQL with pgvector
docker-compose up -d

# Wait 10 seconds for database to start
timeout /t 10

# Initialize database schema
python backend/setup/installer.py --init-db
```

#### Option B: Manual PostgreSQL Installation

See [docs/postgresql_setup.md](docs/postgresql_setup.md) for detailed instructions.

### 6. Verify Installation

```powershell
# Check installation status
python backend/setup/installer.py --check

# You should see:
# ✅ Python Dependencies: OK
# ✅ CLIP Model Cached: Yes
# ✅ PostgreSQL: Connected
# ✅ pgvector Extension: Active
```

## Step-by-Step Installation (Alternative)

If you prefer manual step-by-step installation:

### 1. Install Python Dependencies

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Download AI Model

```powershell
# Download CLIP model (~1.7GB, one-time download)
python backend/setup/installer.py --download-model
```

### 3. Setup PostgreSQL

```powershell
# Show setup guide
python backend/setup/installer.py --setup-postgres

# After PostgreSQL is running:
python backend/setup/installer.py --init-db
```

## Migrating from Old Version (ChromaDB)

If you're upgrading from an older version that used ChromaDB:

```powershell
# Migrate existing data to PostgreSQL
python tools/migrate_to_postgres.py

# Verify migration
python tools/verify_migration.py

# Backup old JSON files
mkdir output\json_backup
move output\json\*.json output\json_backup\
```

## Testing the Installation

### Process a Test Image

```powershell
# Process a single image
python backend/pipeline/ingest_engine.py --file "path\to\test\image.psd"
```

### Search Images

```powershell
# Search using text query
python backend/cli_search_pg.py "cartoon city"

# Hybrid search with filters
python backend/cli_search_pg.py "fantasy character" --mode hybrid --format PSD --min-width 2000
```

### Watch Directory

```powershell
# Auto-process files in a directory
python backend/pipeline/ingest_engine.py --watch "C:\path\to\assets"
```

## Common Issues

### Issue: "connection refused" (PostgreSQL)

**Solution**:
```powershell
# Docker: Check if container is running
docker-compose ps

# Docker: Restart container
docker-compose restart

# Manual: Check PostgreSQL service in services.msc
```

### Issue: "pgvector extension not found"

**Solution**:
```powershell
# Docker: Restart with fresh database
docker-compose down -v
docker-compose up -d
python backend/setup/installer.py --init-db

# Manual: Install pgvector extension
# See docs/postgresql_setup.md
```

### Issue: "CLIP model download stuck"

**Solution**:
```powershell
# Clear Hugging Face cache
rmdir /s /q "%USERPROFILE%\.cache\huggingface"

# Re-download
python backend/setup/installer.py --download-model
```

### Issue: "ImportError: psycopg2"

**Solution**:
```powershell
# Reinstall PostgreSQL adapter
python -m pip install --force-reinstall psycopg2-binary
```

## Directory Structure

After installation, your project should look like:

```
ImageParser/
├── .venv/                      # Python virtual environment
├── backend/
│   ├── db/                     # PostgreSQL client
│   │   ├── schema.sql         # Database schema
│   │   └── pg_client.py       # Python client
│   ├── parser/                # File parsers (PSD, PNG, JPG)
│   ├── pipeline/              # Main processing pipeline
│   ├── search/                # Search API (pgvector)
│   ├── setup/                 # Installation scripts
│   │   └── installer.py       # Main installer ⭐
│   └── vision/                # AI vision analysis
├── tools/
│   ├── migrate_to_postgres.py # Migration script
│   └── verify_migration.py    # Verification script
├── output/
│   └── thumbnails/            # Generated thumbnails
├── docs/
│   └── postgresql_setup.md    # Detailed PostgreSQL guide
├── docker-compose.yml         # PostgreSQL + pgvector (Docker)
├── requirements.txt           # Python dependencies
└── INSTALLATION.md            # This file
```

## Next Steps

After successful installation:

1. **Process your first image**:
   ```powershell
   python backend/pipeline/ingest_engine.py --file "path\to\image.psd"
   ```

2. **Test vector search**:
   ```powershell
   python backend/cli_search_pg.py "your search query"
   ```

3. **Setup directory watching** (optional):
   ```powershell
   python backend/pipeline/ingest_engine.py --watch "C:\path\to\assets"
   ```

4. **Read documentation**:
   - [CLAUDE.md](CLAUDE.md) - Project architecture and development guide
   - [docs/postgresql_setup.md](docs/postgresql_setup.md) - PostgreSQL details
   - [docs/phase_roadmap.md](docs/phase_roadmap.md) - Project roadmap

## Getting Help

- Check [docs/troubleshooting.md](docs/troubleshooting.md) for common issues
- Run diagnostic: `python backend/setup/installer.py --check`
- Open an issue on GitHub (if applicable)

## Cloud Deployment (Optional)

For multi-device synchronization, see:
- [Supabase Guide](docs/postgresql_setup.md#supabase-recommended---500mb-free)
- [Render Guide](docs/postgresql_setup.md#render-free-postgresql)

## Performance Expectations

- **Image processing**: 2-5 seconds per PSD file
- **Vector search**: <50ms for 10,000 images
- **Storage**: ~50KB per image (metadata + vector)
- **CLIP model**: ~1.7GB RAM usage

## System Requirements

- **Minimum**: 4GB RAM, 10GB disk space
- **Recommended**: 8GB RAM, 20GB disk space
- **GPU**: Optional (CUDA for faster CLIP encoding)

---

**Enjoy using ImageParser!** 🎨✨
