"""
v3.1 Schema Migration
=====================

Changes:
1. ai_caption → mc_caption (RENAME COLUMN)
2. Add perceptual_hash, dup_group_id columns + indexes
3. Rebuild FTS5 to 3 columns: meta_strong, meta_weak, caption
4. Backfill FTS with new column structure
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class V31Migration:
    """v3.1 schema migration handler."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent.parent / "imageparser.db")

        self.db_path = db_path
        self.conn = None

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to: {self.db_path}")

    def _get_sqlite_version(self) -> tuple:
        """Get SQLite version as tuple (major, minor, patch)."""
        cursor = self.conn.execute("SELECT sqlite_version()")
        version_str = cursor.fetchone()[0]
        return tuple(map(int, version_str.split('.')))

    def _rename_column_modern(self):
        """Rename ai_caption → mc_caption using ALTER TABLE (SQLite 3.25+)."""
        try:
            self.conn.execute("ALTER TABLE files RENAME COLUMN ai_caption TO mc_caption")
            self.conn.commit()
            logger.info("✅ Renamed ai_caption → mc_caption (modern method)")
            return True
        except Exception as e:
            logger.warning(f"Modern RENAME failed: {e}")
            return False

    def _rename_column_fallback(self):
        """Fallback: add mc_caption, copy data, drop ai_caption (SQLite < 3.25)."""
        try:
            # 1. Add new column
            self.conn.execute("ALTER TABLE files ADD COLUMN mc_caption TEXT")

            # 2. Copy data
            self.conn.execute("UPDATE files SET mc_caption = ai_caption")

            # 3. Drop old column by recreating table (complex, skip for now)
            # For v3.1, we'll leave ai_caption NULL if fallback is needed
            logger.info("✅ Added mc_caption and copied data (fallback method)")
            logger.warning("⚠️ ai_caption column still exists (not dropped in fallback)")
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Fallback RENAME failed: {e}")
            return False

    def step1_rename_ai_caption(self):
        """Step 1: ai_caption → mc_caption."""
        logger.info("=== Step 1: Rename ai_caption → mc_caption ===")

        # Check if already migrated
        cursor = self.conn.execute("PRAGMA table_info(files)")
        cols = {row[1] for row in cursor.fetchall()}

        if 'mc_caption' in cols:
            logger.info("✅ mc_caption already exists, skipping")
            return True

        if 'ai_caption' not in cols:
            logger.error("❌ ai_caption column not found!")
            return False

        # Try modern method first
        version = self._get_sqlite_version()
        logger.info(f"SQLite version: {'.'.join(map(str, version))}")

        if version >= (3, 25, 0):
            if self._rename_column_modern():
                return True

        # Fallback
        logger.warning("Using fallback method (add + copy)")
        return self._rename_column_fallback()

    def step2_add_perceptual_hash_columns(self):
        """Step 2: Add perceptual_hash, dup_group_id columns."""
        logger.info("=== Step 2: Add perceptual_hash columns ===")

        cursor = self.conn.execute("PRAGMA table_info(files)")
        cols = {row[1] for row in cursor.fetchall()}

        added = 0
        if 'perceptual_hash' not in cols:
            self.conn.execute("ALTER TABLE files ADD COLUMN perceptual_hash INTEGER")
            added += 1

        if 'dup_group_id' not in cols:
            self.conn.execute("ALTER TABLE files ADD COLUMN dup_group_id INTEGER")
            added += 1

        if added > 0:
            # Create indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON files(perceptual_hash)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_dup_group_id ON files(dup_group_id)")
            self.conn.commit()
            logger.info(f"✅ Added {added} columns + indexes")
        else:
            logger.info("✅ perceptual_hash columns already exist")

        return True

    def step3_rebuild_fts(self):
        """Step 3: Rebuild FTS5 to 3 columns."""
        logger.info("=== Step 3: Rebuild FTS5 (3 columns) ===")

        # Drop old FTS + triggers
        self.conn.executescript("""
            DROP TRIGGER IF EXISTS files_fts_insert;
            DROP TRIGGER IF EXISTS files_fts_update;
            DROP TRIGGER IF EXISTS files_fts_delete;
            DROP TABLE IF EXISTS files_fts;
        """)

        # Create new FTS with 3 columns
        self.conn.execute("""
            CREATE VIRTUAL TABLE files_fts USING fts5(
                meta_strong,
                meta_weak,
                caption
            )
        """)

        # Create triggers (SQL side sets all columns to '', Python fills later)
        self.conn.executescript("""
            CREATE TRIGGER files_fts_insert AFTER INSERT ON files BEGIN
                INSERT INTO files_fts(rowid, meta_strong, meta_weak, caption)
                VALUES (new.id, '', '', '');
            END;

            CREATE TRIGGER files_fts_update AFTER UPDATE ON files BEGIN
                DELETE FROM files_fts WHERE rowid = old.id;
                INSERT INTO files_fts(rowid, meta_strong, meta_weak, caption)
                VALUES (new.id, '', '', '');
            END;

            CREATE TRIGGER files_fts_delete AFTER DELETE ON files BEGIN
                DELETE FROM files_fts WHERE rowid = old.id;
            END;
        """)

        self.conn.commit()
        logger.info("✅ FTS5 rebuilt (3 columns: meta_strong, meta_weak, caption)")
        return True

    def step4_backfill_fts(self):
        """Step 4: Backfill FTS with existing data."""
        logger.info("=== Step 4: Backfill FTS ===")

        # Use helper methods from this class
        cursor = self.conn.execute(
            "SELECT id, file_path, file_name, mc_caption, ai_tags, "
            "metadata, ocr_text, user_note, user_tags, "
            "image_type, scene_type, art_style, folder_tags FROM files"
        )

        rows_updated = 0
        for row in cursor.fetchall():
            file_id = row[0]
            mc_caption = row[3] or ''
            ai_tags_raw = row[4] or ''
            metadata_str = row[5] or '{}'

            try:
                meta = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                meta = {}

            # Build FTS columns
            meta_strong = self._build_meta_strong(row, meta)
            meta_weak = self._build_meta_weak(row, meta)
            caption = self._build_caption(mc_caption, ai_tags_raw)

            # INSERT (not UPDATE) because we dropped the table
            self.conn.execute(
                "INSERT INTO files_fts(rowid, meta_strong, meta_weak, caption) VALUES (?, ?, ?, ?)",
                (file_id, meta_strong, meta_weak, caption)
            )
            rows_updated += 1

        self.conn.commit()
        logger.info(f"✅ FTS backfilled ({rows_updated} rows)")
        return True

    def _build_meta_strong(self, row, meta: dict) -> str:
        """Build meta_strong: file_name, layer_names, used_fonts, user_tags, ocr_text."""
        parts = []

        # file_name
        parts.append(str(row[1]) if row[1] else '')  # file_name

        # layer_names (from metadata)
        layer_names = self._extract_layer_names(meta)
        if layer_names:
            parts.append(layer_names)

        # used_fonts
        fonts = meta.get('used_fonts', [])
        if isinstance(fonts, list):
            parts.append(' '.join(fonts))

        # user_tags
        user_tags_raw = row[8] or ''
        if user_tags_raw:
            try:
                tags = json.loads(user_tags_raw)
                if isinstance(tags, list):
                    parts.append(' '.join(str(t) for t in tags))
            except:
                pass

        # ocr_text
        parts.append(str(row[6]) if row[6] else '')  # ocr_text

        return ' '.join(str(p) for p in parts if p)

    def _build_meta_weak(self, row, meta: dict) -> str:
        """Build meta_weak: file_path, text_content, user_note, folder_tags, image_type, scene_type, art_style."""
        parts = []

        # file_path
        parts.append(str(row[0]) if row[0] else '')  # file_path

        # text_content (from metadata)
        text_content = meta.get('text_content', [])
        if isinstance(text_content, list):
            parts.append(' '.join(str(t) for t in text_content))

        # user_note
        parts.append(str(row[7]) if row[7] else '')  # user_note

        # folder_tags
        folder_tags_raw = row[12] or ''
        if folder_tags_raw:
            try:
                ft = json.loads(folder_tags_raw)
                if isinstance(ft, list):
                    parts.append(' '.join(str(t) for t in ft))
            except:
                pass

        # image_type, scene_type, art_style
        parts.append(str(row[9]) if row[9] else '')   # image_type
        parts.append(str(row[10]) if row[10] else '')  # scene_type
        parts.append(str(row[11]) if row[11] else '')  # art_style

        return ' '.join(str(p) for p in parts if p)

    def _build_caption(self, mc_caption: str, ai_tags_raw: str) -> str:
        """Build caption: mc_caption + ai_tags."""
        parts = []

        # mc_caption
        if mc_caption:
            parts.append(mc_caption)

        # ai_tags
        if ai_tags_raw:
            try:
                tags = json.loads(ai_tags_raw)
                if isinstance(tags, list):
                    parts.append(' '.join(str(t) for t in tags))
                else:
                    parts.append(str(tags))
            except:
                parts.append(str(ai_tags_raw))

        return ' '.join(p for p in parts if p)

    def _extract_layer_names(self, meta: dict) -> str:
        """Extract layer names from layer_tree."""
        tree = meta.get('layer_tree')
        if not tree or not isinstance(tree, dict):
            return ''

        names = []

        def walk(node):
            name = node.get('name', '')
            if name and name != 'Root':
                names.append(name)
            cleaned = node.get('cleaned_name', '')
            if cleaned and cleaned != name and cleaned != 'Root':
                names.append(cleaned)
            for child in node.get('children', []):
                walk(child)

        walk(tree)

        # Deduplicate
        seen = set()
        unique = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)

        return ' '.join(unique)

    def run(self):
        """Run full migration."""
        try:
            self._connect()

            if not self.step1_rename_ai_caption():
                raise Exception("Step 1 failed")

            if not self.step2_add_perceptual_hash_columns():
                raise Exception("Step 2 failed")

            if not self.step3_rebuild_fts():
                raise Exception("Step 3 failed")

            if not self.step4_backfill_fts():
                raise Exception("Step 4 failed")

            logger.info("🎉 v3.1 migration complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            if self.conn:
                self.conn.rollback()
            return False

        finally:
            if self.conn:
                self.conn.close()


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    migration = V31Migration()
    success = migration.run()

    if success:
        print("\n[OK] Migration successful!")
        print("You can now use mc_caption, perceptual_hash, and the new 3-column FTS.")
    else:
        print("\n[ERROR] Migration failed. Check logs above.")
        exit(1)


if __name__ == "__main__":
    main()
