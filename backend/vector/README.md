# Legacy ChromaDB Module (Deprecated)

**⚠️ This module is deprecated and kept for backward compatibility only.**

## Migration Notice

ImageParser has migrated from **ChromaDB** to **PostgreSQL + pgvector** for better performance and unified storage.

### What Changed

**Before (ChromaDB + JSON files)**:
- Metadata stored in: `output/json/*.json`
- Vectors stored in: `chroma_db/` (ChromaDB)
- Search speed: ~1.2 seconds (194 files)
- Nested JSON: Not supported (ChromaDB limitation)

**After (PostgreSQL + pgvector)**:
- Metadata + Vectors: Single PostgreSQL database
- Search speed: ~20ms (60x faster!)
- Nested JSON: Fully supported (JSONB)
- Scalable: Handles 10,000+ images

### Migration Guide

If you have existing ChromaDB data:

```powershell
# Migrate to PostgreSQL
python tools/migrate_to_postgres.py

# Verify migration
python tools/verify_migration.py

# After verification, you can safely remove ChromaDB
```

### Files in This Directory

| File | Status | Purpose |
|------|--------|---------|
| `indexer.py` | 🔴 Deprecated | ChromaDB vector indexing (replaced by `backend/db/pg_client.py`) |
| `searcher.py` | 🔴 Deprecated | ChromaDB vector search (replaced by `backend/search/pg_search.py`) |
| `README.md` | ℹ️ This file | Migration documentation |

### New Implementation

Use these modules instead:

| Old Module | New Module | Purpose |
|------------|------------|---------|
| `backend.vector.indexer.VectorIndexer` | `backend.db.pg_client.PostgresDB` | Store metadata + vectors |
| `backend.vector.searcher.VectorSearcher` | `backend.search.pg_search.PgVectorSearch` | Search images |

### Code Migration Example

**Before (ChromaDB)**:
```python
from backend.vector.indexer import VectorIndexer
from backend.vector.searcher import VectorSearcher

indexer = VectorIndexer()
indexer.index_image(file_path, metadata, thumbnail_path)

searcher = VectorSearcher()
results = searcher.search("cartoon city")
```

**After (PostgreSQL)**:
```python
from backend.db.pg_client import PostgresDB
from backend.search.pg_search import PgVectorSearch
from sentence_transformers import SentenceTransformer

# Storage
db = PostgresDB()
clip_model = SentenceTransformer('clip-ViT-L-14')
embedding = clip_model.encode(image)
db.insert_file(file_path, metadata, embedding)

# Search
search = PgVectorSearch(db=db)
results = search.vector_search("cartoon city", top_k=20)
```

### Why We Migrated

1. **Performance**: PostgreSQL is 60x faster for search
2. **Unified Storage**: Metadata + vectors in one database
3. **JSONB Support**: Full nested JSON support (layer_tree)
4. **Hybrid Search**: Vector + metadata filters in single query
5. **Scalability**: Better performance with 10,000+ images
6. **Cloud Ready**: Easy deployment on Supabase, Render, etc.

### Removing ChromaDB

After successful migration and verification:

```powershell
# 1. Backup ChromaDB data (optional)
mkdir backup
xcopy chroma_db backup\chroma_db /E /I

# 2. Remove ChromaDB from requirements.txt
# (Edit requirements.txt and remove line: chromadb>=0.4.0)

# 3. Uninstall ChromaDB (optional)
python -m pip uninstall chromadb -y

# 4. Delete ChromaDB directory (after 1 month backup period)
rmdir /s /q chroma_db
```

### Keep or Delete?

**Keep for now (1 month backup period)**:
- `chroma_db/` directory (backup)
- `output/json/` files (backup)
- `backend/vector/` module (for migration script compatibility)

**Safe to delete after verification**:
- Individual JSON files in `output/json/` (after moving to `output/json_backup/`)
- ChromaDB pip package
- `chroma_db/` directory

**Do not delete** (still used):
- `output/thumbnails/` - Still used for image previews

## FAQ

**Q: Can I still use ChromaDB?**
A: Yes, for backward compatibility. But new installations should use PostgreSQL.

**Q: Will old code break?**
A: No. Migration script (`migrate_to_postgres.py`) handles data transfer. After migration, update your code to use new modules.

**Q: What if migration fails?**
A: Your JSON files and ChromaDB data are preserved. You can retry migration or continue using ChromaDB temporarily.

**Q: Can I use both systems?**
A: Not recommended. Choose one:
- **New projects**: PostgreSQL + pgvector
- **Existing projects**: Migrate to PostgreSQL for better performance

**Q: How do I uninstall ChromaDB completely?**
A: After successful PostgreSQL migration:
```powershell
python tools/verify_migration.py  # Ensure data is correct
python -m pip uninstall chromadb -y
rmdir /s /q chroma_db
```

## Support

For migration help:
- Check [INSTALLATION.md](../../INSTALLATION.md)
- Check [docs/postgresql_setup.md](../../docs/postgresql_setup.md)
- Run migration script: `python tools/migrate_to_postgres.py`
- Run verification: `python tools/verify_migration.py`
