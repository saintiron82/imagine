import sqlite3
import os
import sys
from pathlib import Path

def migrate():
    """Add user metadata columns to files table."""
    db_path = os.getenv('SQLITE_DB_PATH', 'imageparser.db')

    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        print("Please run the ingest pipeline first to create the database.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns exist
        cursor.execute("PRAGMA table_info(files)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'user_note' not in columns:
            print("[*] Adding user metadata columns...")
            cursor.execute("ALTER TABLE files ADD COLUMN user_note TEXT DEFAULT ''")
            cursor.execute("ALTER TABLE files ADD COLUMN user_tags TEXT DEFAULT '[]'")
            cursor.execute("ALTER TABLE files ADD COLUMN user_category TEXT DEFAULT ''")
            cursor.execute("ALTER TABLE files ADD COLUMN user_rating INTEGER DEFAULT 0")
            conn.commit()
            print("[OK] User metadata columns added successfully")

            # Verify
            cursor.execute("PRAGMA table_info(files)")
            new_columns = [col[1] for col in cursor.fetchall()]
            added = [c for c in ['user_note', 'user_tags', 'user_category', 'user_rating'] if c in new_columns]
            print(f"    Added columns: {', '.join(added)}")
        else:
            print("[INFO] User metadata columns already exist - no changes needed")

        # Update FTS5 table to include user fields
        print("\n[*] Updating FTS5 index to include user metadata...")

        # Check if FTS5 table exists and has user fields
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='files_fts'")
        fts_schema = cursor.fetchone()

        if fts_schema and 'user_note' not in fts_schema[0]:
            print("    Recreating FTS5 table with user metadata...")

            # Drop existing FTS5 table and triggers
            cursor.execute("DROP TABLE IF EXISTS files_fts")
            cursor.execute("DROP TRIGGER IF EXISTS files_fts_insert")
            cursor.execute("DROP TRIGGER IF EXISTS files_fts_update")
            cursor.execute("DROP TRIGGER IF EXISTS files_fts_delete")

            # Recreate FTS5 table with user fields
            cursor.execute("""
                CREATE VIRTUAL TABLE files_fts USING fts5(
                    file_path,
                    ai_caption,
                    semantic_tags,
                    user_note,
                    user_tags
                )
            """)

            # Recreate triggers
            cursor.execute("""
                CREATE TRIGGER files_fts_insert AFTER INSERT ON files BEGIN
                    INSERT INTO files_fts(rowid, file_path, ai_caption, semantic_tags, user_note, user_tags)
                    VALUES (new.id, new.file_path, new.ai_caption,
                            json_extract(new.metadata, '$.semantic_tags'),
                            new.user_note, new.user_tags);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER files_fts_update AFTER UPDATE ON files BEGIN
                    DELETE FROM files_fts WHERE rowid = old.id;
                    INSERT INTO files_fts(rowid, file_path, ai_caption, semantic_tags, user_note, user_tags)
                    VALUES (new.id, new.file_path, new.ai_caption,
                            json_extract(new.metadata, '$.semantic_tags'),
                            new.user_note, new.user_tags);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER files_fts_delete AFTER DELETE ON files BEGIN
                    DELETE FROM files_fts WHERE rowid = old.id;
                END
            """)

            # Repopulate FTS5 table from existing data
            cursor.execute("""
                INSERT INTO files_fts(rowid, file_path, ai_caption, semantic_tags, user_note, user_tags)
                SELECT id, file_path, ai_caption,
                       json_extract(metadata, '$.semantic_tags'),
                       COALESCE(user_note, ''), COALESCE(user_tags, '[]')
                FROM files
            """)

            conn.commit()
            print("[OK] FTS5 index updated successfully")
        else:
            print("[INFO] FTS5 index already includes user metadata")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == '__main__':
    print("=== User Metadata Migration ===")
    migrate()
    print("\n[OK] Migration completed successfully!")
