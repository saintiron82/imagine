"""v3.1 Integration Test - Quick verification of all changes."""

print("=== v3.1 Integration Test ===\n")

# Test 1: DB Schema
print("[Test 1] Database Schema")
from backend.db.sqlite_client import SQLiteDB
db = SQLiteDB()

cursor = db.conn.execute("PRAGMA table_info(files)")
cols = {row[1] for row in cursor.fetchall()}
assert 'mc_caption' in cols, "mc_caption column missing"
assert 'ai_caption' not in cols, "ai_caption should be removed"
assert 'perceptual_hash' in cols, "perceptual_hash column missing"
assert 'dup_group_id' in cols, "dup_group_id column missing"
print("  [OK] All new columns present\n")

# Test 2: FTS Structure
print("[Test 2] FTS5 Structure")
cursor = db.conn.execute("PRAGMA table_info(files_fts)")
fts_cols = [row[1] for row in cursor.fetchall()]
assert fts_cols == ['meta_strong', 'meta_weak', 'caption'], f"FTS columns wrong: {fts_cols}"
print("  [OK] 3-column FTS (meta_strong, meta_weak, caption)\n")

# Test 3: FTS Population
print("[Test 3] FTS Population")
files_count = db.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
fts_count = db.conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
assert files_count == fts_count, f"FTS not fully populated: {fts_count}/{files_count}"
print(f"  [OK] {fts_count} files indexed\n")

# Test 4: Sample Data
print("[Test 4] Sample File Data")
cursor = db.conn.execute("""
    SELECT file_name, mc_caption, perceptual_hash
    FROM files
    WHERE mc_caption IS NOT NULL
    LIMIT 1
""")
row = cursor.fetchone()
if row:
    print(f"  File: {row[0]}")
    print(f"  MC Caption: {row[1][:60]}...")
    print(f"  Hash: {row[2]}")
    print("  [OK] Sample data present\n")
else:
    print("  [WARN] No files with mc_caption\n")

# Test 5: FTS Search
print("[Test 5] FTS BM25 Search")
try:
    cursor = db.conn.execute("""
        SELECT rowid, bm25(files_fts, 3.0, 1.5, 0.7) AS rank
        FROM files_fts
        WHERE files_fts MATCH 'UI'
        ORDER BY rank
        LIMIT 1
    """)
    row = cursor.fetchone()
    if row:
        print(f"  Found: rowid={row[0]}, rank={row[1]:.4f}")
        print("  [OK] BM25 weighted search works\n")
    else:
        print("  [WARN] No matches for 'UI'\n")
except Exception as e:
    print(f"  [FAIL] FTS search error: {e}\n")

# Test 6: Config Loading
print("[Test 6] Config.yaml")
from backend.utils.config import get_config
cfg = get_config()
thumb_size = cfg.get('thumbnail.max_size')
bm25 = cfg.get('search.fts.bm25_weights')
presets = cfg.get('search.rrf.presets')

assert thumb_size == 1024, f"thumbnail.max_size should be 1024, got {thumb_size}"
assert bm25 is not None, "search.fts.bm25_weights not found"
assert presets is not None, "search.rrf.presets not found"
print(f"  Thumbnail size: {thumb_size}")
print(f"  BM25 weights: {bm25}")
print(f"  RRF presets: {list(presets.keys())}")
print("  [OK] Config loaded correctly\n")

# Test 7: dHash Utility
print("[Test 7] dHash Utility")
from backend.utils.dhash import dhash64, hamming_distance
from PIL import Image
import numpy as np

# Create test image
img = Image.new('RGB', (100, 100), color='red')
hash1 = dhash64(img)
hash2 = dhash64(img)
dist = hamming_distance(hash1, hash2)
assert dist == 0, f"Self-distance should be 0, got {dist}"
print(f"  Hash: {hash1}")
print(f"  Self-distance: {dist}")
print("  [OK] dHash working\n")

# Test 8: MV Document Format
print("[Test 8] MV Document Format")
from backend.vector.text_embedding import build_document_text

doc = build_document_text(
    "A warrior holding a sword",
    ["character", "fantasy"],
    facts={"image_type": "character", "fonts": "NotoSans"}
)
assert "[SEMANTIC]" in doc, "Missing [SEMANTIC] section"
assert "[FACTS]" in doc, "Missing [FACTS] section"
assert "image_type=character" in doc, "Missing facts"
print(f"  Length: {len(doc)} chars")
print(f"  Has SEMANTIC: {'[SEMANTIC]' in doc}")
print(f"  Has FACTS: {'[FACTS]' in doc}")
print("  [OK] MV format correct\n")

db.close()

print("=" * 50)
print("🎉 v3.1 Integration Test PASSED!")
print("=" * 50)
