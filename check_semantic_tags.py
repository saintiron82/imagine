"""Check semantic_tags statistics in database."""
from backend.db.sqlite_client import SQLiteDB
import json

db = SQLiteDB()
cursor = db.conn.execute('SELECT file_name, format, metadata FROM files')
rows = cursor.fetchall()

total = 0
empty = 0
has_tags = 0
samples_empty = []
samples_tags = []

for row in rows:
    file_name, fmt, metadata_json = row
    if metadata_json:
        total += 1
        meta = json.loads(metadata_json)
        tags = meta.get('semantic_tags', '')

        if not tags or tags.strip() == '':
            empty += 1
            if len(samples_empty) < 5:
                samples_empty.append((file_name, fmt))
        else:
            has_tags += 1
            if len(samples_tags) < 5:
                samples_tags.append((file_name, fmt, tags[:50]))

print(f'Total files: {total}')
print(f'Empty semantic_tags: {empty} ({empty*100//total if total > 0 else 0}%)')
print(f'Has semantic_tags: {has_tags} ({has_tags*100//total if total > 0 else 0}%)')

print('\nSamples with EMPTY semantic_tags:')
for fname, fmt in samples_empty:
    print(f'  {fname} ({fmt})')

print('\nSamples with semantic_tags:')
for fname, fmt, tags in samples_tags:
    print(f'  {fname} ({fmt}): "{tags}..."')
