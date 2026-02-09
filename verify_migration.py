"""Verify v3.1 migration results."""

import sqlite3

conn = sqlite3.connect('imageparser.db')

# Check columns
print("=== Files table columns ===")
cursor = conn.execute("PRAGMA table_info(files)")
cols = {row[1] for row in cursor.fetchall()}

checks = {
    'mc_caption': 'mc_caption' in cols,
    'ai_caption_removed': 'ai_caption' not in cols,
    'perceptual_hash': 'perceptual_hash' in cols,
    'dup_group_id': 'dup_group_id' in cols,
}

for name, passed in checks.items():
    status = '[OK]' if passed else '[FAIL]'
    print(f"{status} {name}")

# Check FTS structure
print("\n=== FTS5 structure ===")
cursor = conn.execute("PRAGMA table_info(files_fts)")
fts_cols = [row[1] for row in cursor.fetchall()]
print(f"FTS columns: {fts_cols}")

expected_fts = ['meta_strong', 'meta_weak', 'caption']
fts_ok = fts_cols == expected_fts
print(f"[{'OK' if fts_ok else 'FAIL'}] FTS has {len(fts_cols)} columns (expected 3)")

# Check FTS population
cursor = conn.execute("SELECT COUNT(*) FROM files")
files_count = cursor.fetchone()[0]

cursor = conn.execute("SELECT COUNT(*) FROM files_fts")
fts_count = cursor.fetchone()[0]

print(f"\n=== FTS population ===")
print(f"Files: {files_count}")
print(f"FTS rows: {fts_count}")
print(f"[{'OK' if files_count == fts_count else 'FAIL'}] FTS fully populated")

# Sample FTS content
print(f"\n=== Sample FTS content ===")
cursor = conn.execute("SELECT rowid, meta_strong, meta_weak, caption FROM files_fts LIMIT 1")
row = cursor.fetchone()
if row:
    print(f"Row ID: {row[0]}")
    print(f"meta_strong: {row[1][:100]}..." if len(row[1]) > 100 else f"meta_strong: {row[1]}")
    print(f"meta_weak: {row[2][:100]}..." if len(row[2]) > 100 else f"meta_weak: {row[2]}")
    print(f"caption: {row[3][:100]}..." if len(row[3]) > 100 else f"caption: {row[3]}")

conn.close()
print("\n=== Migration verification complete ===")
