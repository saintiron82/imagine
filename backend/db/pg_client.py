"""
PostgreSQL client with pgvector support.

This module provides a wrapper around psycopg2 for interacting with the
PostgreSQL database, including:
- Schema initialization
- File metadata + CLIP vector storage
- Layer metadata storage
- CRUD operations
- Connection management
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import numpy as np

logger = logging.getLogger(__name__)


class PostgresDB:
    """PostgreSQL database client with pgvector support."""

    def __init__(self, conn_string: Optional[str] = None):
        """
        Initialize PostgreSQL connection.

        Args:
            conn_string: PostgreSQL connection string.
                        Default: postgresql://postgres:password@localhost:5432/imageparser
        """
        if conn_string is None:
            conn_string = "postgresql://postgres:password@localhost:5432/imageparser"

        self.conn_string = conn_string
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.conn_string)
            self.conn.autocommit = False  # Use transactions
            logger.info(f"✅ Connected to PostgreSQL")
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.error("Please ensure PostgreSQL is running and pgvector extension is installed.")
            logger.error("Install guide: https://github.com/pgvector/pgvector")
            raise

    def init_schema(self):
        """
        Initialize database schema from schema.sql file.
        Creates tables and indexes if they don't exist.
        """
        cursor = self.conn.cursor()

        try:
            # Read schema.sql
            schema_path = Path(__file__).parent / "schema.sql"
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")

            with open(schema_path, encoding='utf-8') as f:
                schema_sql = f.read()

            # Execute schema
            cursor.execute(schema_sql)
            self.conn.commit()

            logger.info("✅ PostgreSQL schema initialized successfully")

            # Verify extensions
            cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');")
            extensions = [row[0] for row in cursor.fetchall()]

            if 'vector' not in extensions:
                logger.warning("⚠️ pgvector extension not found! Vector search will not work.")
                logger.warning("Install: https://github.com/pgvector/pgvector")
            else:
                logger.info("✅ pgvector extension is active")

            if 'pg_trgm' not in extensions:
                logger.warning("⚠️ pg_trgm extension not found! Text similarity search may be limited.")
            else:
                logger.info("✅ pg_trgm extension is active")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Schema initialization failed: {e}")
            raise
        finally:
            cursor.close()

    def insert_file(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray
    ) -> int:
        """
        Insert or update file metadata + CLIP vector.

        Args:
            file_path: Absolute file path (unique identifier)
            metadata: Full metadata dict from AssetMeta.model_dump()
            embedding: CLIP embedding vector (768 dimensions)

        Returns:
            Database ID of inserted/updated record
        """
        cursor = self.conn.cursor()

        try:
            # Extract nested data for JSONB storage
            metadata_jsonb = {
                "layer_tree": metadata.get("layer_tree"),
                "translated_layer_tree": metadata.get("translated_layer_tree"),
                "translated_layer_tree_en": metadata.get("translated_layer_tree_en"),
                "semantic_tags": metadata.get("semantic_tags"),
                "translated_tags": metadata.get("translated_tags"),
                "text_content": metadata.get("text_content"),
                "layer_count": metadata.get("layer_count"),
                "used_fonts": metadata.get("used_fonts"),
            }

            # Extract resolution tuple
            resolution = metadata.get("resolution", (None, None))
            width = resolution[0] if isinstance(resolution, (list, tuple)) else None
            height = resolution[1] if isinstance(resolution, (list, tuple)) else None

            cursor.execute("""
                INSERT INTO files (
                    file_path, file_name, file_size, format, width, height,
                    ai_caption, ai_tags, ocr_text, dominant_color, ai_style,
                    metadata, thumbnail_url, embedding,
                    created_at, modified_at, parsed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, NOW()
                )
                ON CONFLICT (file_path) DO UPDATE SET
                    ai_caption = EXCLUDED.ai_caption,
                    ai_tags = EXCLUDED.ai_tags,
                    ocr_text = EXCLUDED.ocr_text,
                    dominant_color = EXCLUDED.dominant_color,
                    ai_style = EXCLUDED.ai_style,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    parsed_at = NOW()
                RETURNING id
            """, (
                file_path,
                metadata.get("file_name"),
                metadata.get("file_size"),
                metadata.get("format"),
                width,
                height,
                metadata.get("ai_caption"),
                metadata.get("ai_tags"),
                metadata.get("ocr_text"),
                metadata.get("dominant_color"),
                metadata.get("ai_style"),
                Json(metadata_jsonb),
                metadata.get("thumbnail_url"),
                embedding.tolist(),
                metadata.get("created_at"),
                metadata.get("modified_at"),
            ))

            file_id = cursor.fetchone()[0]
            self.conn.commit()

            logger.debug(f"✅ Indexed file to PostgreSQL: {file_path} (ID: {file_id})")
            return file_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Failed to insert file {file_path}: {e}")
            raise
        finally:
            cursor.close()

    def insert_layer(
        self,
        file_id: int,
        layer_path: str,
        layer_metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> int:
        """
        Insert or update layer metadata + embedding.

        Args:
            file_id: Parent file database ID
            layer_path: Layer path (e.g., "Root/Group 1/Layer 2")
            layer_metadata: Layer properties dict
            embedding: Optional CLIP embedding for this layer

        Returns:
            Database ID of inserted/updated layer
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO layers (
                    file_id, layer_path, layer_name, layer_type,
                    metadata, ai_caption, ai_tags, embedding
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (file_id, layer_path) DO UPDATE SET
                    layer_name = EXCLUDED.layer_name,
                    layer_type = EXCLUDED.layer_type,
                    metadata = EXCLUDED.metadata,
                    ai_caption = EXCLUDED.ai_caption,
                    ai_tags = EXCLUDED.ai_tags,
                    embedding = EXCLUDED.embedding
                RETURNING id
            """, (
                file_id,
                layer_path,
                layer_metadata.get("name"),
                layer_metadata.get("kind"),
                Json(layer_metadata),
                layer_metadata.get("ai_caption"),
                layer_metadata.get("ai_tags"),
                embedding.tolist() if embedding is not None else None,
            ))

            layer_id = cursor.fetchone()[0]
            self.conn.commit()

            logger.debug(f"✅ Indexed layer: {layer_path} (ID: {layer_id})")
            return layer_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Failed to insert layer {layer_path}: {e}")
            raise
        finally:
            cursor.close()

    def get_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file metadata by path.

        Args:
            file_path: Absolute file path

        Returns:
            File record as dict, or None if not found
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("""
                SELECT
                    id, file_path, file_name, file_size, format,
                    width, height, ai_caption, ai_tags, ocr_text,
                    dominant_color, ai_style, metadata, thumbnail_url,
                    created_at, modified_at, parsed_at,
                    embedding::text as embedding_text
                FROM files
                WHERE file_path = %s
            """, (file_path,))

            row = cursor.fetchone()
            return dict(row) if row else None

        finally:
            cursor.close()

    def count_files(self) -> int:
        """Count total files in database."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM files")
            return cursor.fetchone()[0]
        finally:
            cursor.close()

    def count_layers(self) -> int:
        """Count total layers in database."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM layers")
            return cursor.fetchone()[0]
        finally:
            cursor.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        try:
            stats = {}

            # File count
            cursor.execute("SELECT COUNT(*) FROM files")
            stats['total_files'] = cursor.fetchone()[0]

            # Layer count
            cursor.execute("SELECT COUNT(*) FROM layers")
            stats['total_layers'] = cursor.fetchone()[0]

            # Files with AI captions
            cursor.execute("SELECT COUNT(*) FROM files WHERE ai_caption IS NOT NULL")
            stats['files_with_ai_caption'] = cursor.fetchone()[0]

            # Average layers per file
            cursor.execute("""
                SELECT AVG(layer_count)::integer
                FROM files
                WHERE (metadata->>'layer_count')::integer > 0
            """)
            result = cursor.fetchone()[0]
            stats['avg_layers_per_file'] = result if result else 0

            # Format distribution
            cursor.execute("""
                SELECT format, COUNT(*) as count
                FROM files
                GROUP BY format
                ORDER BY count DESC
            """)
            stats['format_distribution'] = dict(cursor.fetchall())

            return stats

        finally:
            cursor.close()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
