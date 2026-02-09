"""Quick check of perceptual_hash in DB."""
from backend.db.sqlite_client import SQLiteDB

db = SQLiteDB()

# Count statistics
cursor = db.conn.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(perceptual_hash) as with_hash,
        COUNT(*) - COUNT(perceptual_hash) as null_count
    FROM files
""")
stats = cursor.fetchone()
print(f"Total files: {stats[0]}")
print(f"With hash: {stats[1]}")
print(f"NULL hash: {stats[2]}")

# Sample values
print("\nSample hashes:")
cursor = db.conn.execute("""
    SELECT file_name, perceptual_hash
    FROM files
    WHERE perceptual_hash IS NOT NULL
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

# Check for duplicate groups
cursor = db.conn.execute("""
    SELECT perceptual_hash, COUNT(*) as cnt, GROUP_CONCAT(file_name, ', ') as files
    FROM files
    WHERE perceptual_hash IS NOT NULL
    GROUP BY perceptual_hash
    HAVING cnt > 1
    LIMIT 5
""")
print("\nDuplicate groups:")
dups = cursor.fetchall()
if dups:
    for row in dups:
        print(f"  Hash {row[0]}: {row[1]} files - {row[2][:100]}...")
else:
    print("  (no duplicates found)")
