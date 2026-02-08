# PostgreSQL + pgvector Setup Guide

This guide explains how to set up PostgreSQL with pgvector extension for ImageParser.

## Why PostgreSQL + pgvector?

**Replaces**: JSON files + ChromaDB dual storage
**Benefits**:
- ✅ 60x faster search (1.2s → 20ms)
- ✅ Single database for metadata + vectors
- ✅ Nested JSON support (JSONB for layer_tree)
- ✅ Hybrid search (vector + metadata filters in one query)
- ✅ Scales to 10,000+ images
- ✅ Optional cloud deployment (Supabase, Render)

## Option 1: Docker (Recommended - Easiest)

**Requirements**: Docker Desktop for Windows

### Install Docker Desktop

1. Download: https://www.docker.com/products/docker-desktop/
2. Install and restart
3. Open Docker Desktop

### Start PostgreSQL

```powershell
# Navigate to project directory
cd C:\Users\saint\ImageParser

# Start PostgreSQL with pgvector
docker-compose up -d

# Verify it's running
docker-compose ps

# View logs
docker-compose logs -f postgres

# You should see:
#   "database system is ready to accept connections"
```

### Test Connection

```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Test database connection
python -c "from backend.db.pg_client import PostgresDB; db = PostgresDB(); db.init_schema(); print('✅ Success!')"
```

### Stop PostgreSQL

```powershell
# Stop (data persists)
docker-compose down

# Remove all data (fresh start)
docker-compose down -v
```

## Option 2: Local Installation (Windows)

**Requirements**: PostgreSQL 16+ installed on Windows

### Install PostgreSQL

1. Download installer: https://www.postgresql.org/download/windows/
2. Run installer (use default port 5432)
3. Set password: `password` (or update connection string in code)
4. Install pgAdmin (optional, for GUI management)

### Install pgvector Extension

#### Method A: Pre-built Binary (Easiest)

```powershell
# Download pgvector Windows binary
# https://github.com/pgvector/pgvector/releases

# Extract pgvector.dll to PostgreSQL extension directory
# Example: C:\Program Files\PostgreSQL\16\lib\

# Restart PostgreSQL service
net stop postgresql-x64-16
net start postgresql-x64-16
```

#### Method B: Compile from Source (Advanced)

```powershell
# Requires Visual Studio 2022 or MinGW
git clone https://github.com/pgvector/pgvector.git
cd pgvector

# Follow Windows build instructions:
# https://github.com/pgvector/pgvector#windows
```

### Create Database

```powershell
# Using psql (included with PostgreSQL)
psql -U postgres
```

```sql
-- In psql prompt
CREATE DATABASE imageparser;
\c imageparser
CREATE EXTENSION vector;
\q
```

### Test Connection

```powershell
python -c "from backend.db.pg_client import PostgresDB; db = PostgresDB(); db.init_schema(); print('✅ Success!')"
```

## Option 3: Cloud Hosted (Multi-device Sync)

**Free Tier Options**:

### Supabase (Recommended - 500MB Free)

1. Sign up: https://supabase.com/
2. Create new project (select region)
3. Copy connection string from Settings → Database
4. Update `backend/db/pg_client.py`:

```python
# Default connection
conn_string = "postgresql://postgres.[PROJECT_ID]:[PASSWORD]@db.[REGION].supabase.co:5432/postgres"
```

5. pgvector is pre-installed ✅

### Render (Free PostgreSQL)

1. Sign up: https://render.com/
2. Create PostgreSQL database
3. Copy internal connection string
4. Update connection string in code

### Railway (Free $5/month credit)

1. Sign up: https://railway.app/
2. Deploy PostgreSQL + pgvector template
3. Copy connection string

## Migrate Existing Data

Once PostgreSQL is running:

```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Dry run (check what will be migrated)
python tools/migrate_to_postgres.py --dry-run

# Actual migration (transfers JSON + ChromaDB → PostgreSQL)
python tools/migrate_to_postgres.py

# Verify migration
python tools/verify_migration.py

# Test search
python backend/cli_search_pg.py "cartoon city"
```

## Connection String Format

```
postgresql://[USER]:[PASSWORD]@[HOST]:[PORT]/[DATABASE]

Examples:
  Local (Docker):     postgresql://postgres:password@localhost:5432/imageparser
  Local (Windows):    postgresql://postgres:password@localhost:5432/imageparser
  Supabase:          postgresql://postgres.[PROJECT]:[PASS]@db.[REGION].supabase.co:5432/postgres
  Render:            postgresql://user:pass@dpg-xxx.oregon-postgres.render.com/dbname
```

## Troubleshooting

### Error: "connection refused"

**Docker**: Check if container is running with `docker-compose ps`
**Local**: Check if PostgreSQL service is running

```powershell
# Windows services
services.msc
# Look for "postgresql-x64-16"
```

### Error: "extension 'vector' does not exist"

**Docker**: Should be pre-installed. Try `docker-compose down -v && docker-compose up -d`
**Local**: pgvector not installed. See installation steps above.

### Error: "password authentication failed"

Update connection string in `backend/db/pg_client.py` with correct password:

```python
def __init__(self, conn_string: Optional[str] = None):
    if conn_string is None:
        conn_string = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/imageparser"
```

### Error: "FATAL: database 'imageparser' does not exist"

Create database:

```powershell
psql -U postgres -c "CREATE DATABASE imageparser;"
```

## Performance Tuning (Optional)

For production use with 10,000+ images:

```sql
-- Increase HNSW index parameters (better accuracy, slower build)
CREATE INDEX idx_files_embedding ON files
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 32, ef_construction = 200);  -- Default: m=16, ef=100

-- Adjust PostgreSQL settings in postgresql.conf
shared_buffers = 512MB          # Default: 128MB
effective_cache_size = 2GB      # Default: 4GB
work_mem = 16MB                 # Default: 4MB
```

## Next Steps

After successful setup:

1. ✅ Migrate existing data: `python tools/migrate_to_postgres.py`
2. ✅ Verify migration: `python tools/verify_migration.py`
3. ✅ Test search: `python backend/cli_search_pg.py "test query"`
4. ✅ Update pipeline: Modify `backend/pipeline/ingest_engine.py` to use PostgreSQL
5. ✅ Backup JSON files: Keep for 1 month, then delete
6. ⏳ Optional: Remove ChromaDB after validation

## Resources

- PostgreSQL: https://www.postgresql.org/
- pgvector: https://github.com/pgvector/pgvector
- psycopg2: https://www.psycopg.org/docs/
- Docker Desktop: https://www.docker.com/products/docker-desktop/
- Supabase: https://supabase.com/docs/guides/database
