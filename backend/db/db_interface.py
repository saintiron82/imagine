"""
Database interface (abstract protocol) for Imagine.

Both SQLiteDB and PostgresDB implement this protocol,
allowing transparent backend switching via db_factory.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class DBInterface(ABC):
    """Abstract database interface for Imagine storage layer.

    All concrete DB clients (SQLiteDB, PostgresDB) must implement
    these methods with identical signatures and semantics.
    """

    # ── Phase 1: Parse ──────────────────────────────────────────

    @abstractmethod
    def upsert_metadata(self, file_path: str, metadata: Dict[str, Any], commit: bool = True) -> int:
        """INSERT basic metadata, preserve existing AI fields on conflict.

        Returns database file ID.
        """
        ...

    @abstractmethod
    def insert_layer(
        self, file_id: int, layer_path: str, layer_name: str,
        layer_type: str = None, metadata: Dict = None,
        ai_caption: str = None, ai_tags: List[str] = None,
        commit: bool = True
    ) -> int:
        """Insert or update a layer record for a file.

        Returns layer ID.
        """
        ...

    # ── Phase 2: Vision ─────────────────────────────────────────

    @abstractmethod
    def update_vision_fields(self, file_path: str, fields: Dict[str, Any], commit: bool = True) -> bool:
        """UPDATE only VLM-generated fields (mc_caption, ai_tags, classification, etc.).

        Returns True if a row was updated.
        """
        ...

    # ── Phase 3: Embed ──────────────────────────────────────────

    @abstractmethod
    def upsert_vectors(self, file_id: int, vv_vec=None, mv_vec=None, structure_vec=None, commit: bool = True) -> bool:
        """INSERT/REPLACE VV, MV, and Structure vectors.

        Args:
            file_id: Database file ID (from upsert_metadata)
            vv_vec: numpy array for VV (Visual Vector), or None to skip
            mv_vec: numpy array for MV (Meaning Vector), or None to skip
            structure_vec: numpy array for Structure Vector (DINOv2), or None to skip

        Returns True on success.
        """
        ...

    # ── Query: Single file ──────────────────────────────────────

    @abstractmethod
    def get_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get full file metadata by path. Returns None if not found."""
        ...

    @abstractmethod
    def get_file_modified_at(self, file_path: str) -> Optional[str]:
        """Get stored modified_at timestamp for a file."""
        ...

    @abstractmethod
    def get_file_mode_tier(self, file_path: str) -> Optional[str]:
        """Get stored mode_tier for a file."""
        ...

    @abstractmethod
    def get_file_phase_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get per-phase completion info for smart skip.

        Returns dict with:
            file_id, caption_model, embedding_model, text_embed_model, mode_tier,
            has_mc, has_vv, has_mv, has_structure, modified_at, file_size, content_hash
        """
        ...

    @abstractmethod
    def find_by_content_hash(self, content_hash: str) -> List[Dict[str, Any]]:
        """Find files by content_hash. Returns list of dicts."""
        ...

    # ── Query: Aggregates ───────────────────────────────────────

    @abstractmethod
    def count_files(self) -> int:
        """Count total files in database."""
        ...

    @abstractmethod
    def count_layers(self) -> int:
        """Count total layers in database."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics (total files, layers, MC/VV/MV counts, format distribution)."""
        ...

    @abstractmethod
    def get_incomplete_stats(self) -> Dict[str, Any]:
        """Get incomplete file stats grouped by storage_root."""
        ...

    @abstractmethod
    def get_folder_phase_stats(self, root_path: str) -> List[Dict[str, Any]]:
        """Get per-storage_root phase completion stats under root_path prefix."""
        ...

    @abstractmethod
    def get_build_status(self) -> Dict[str, Any]:
        """Get data-build compatibility status for rebuild guidance."""
        ...

    # ── Mutation: User metadata ─────────────────────────────────

    @abstractmethod
    def update_user_metadata(
        self,
        file_path: str,
        user_note: Optional[str] = None,
        user_tags: Optional[List[str]] = None,
        user_category: Optional[str] = None,
        user_rating: Optional[int] = None
    ) -> bool:
        """Update user metadata (note, tags, category, rating) for a file."""
        ...

    # ── Mutation: Delete / Relink ───────────────────────────────

    @abstractmethod
    def delete_file(self, file_id: int) -> bool:
        """Delete a file row by ID (cascades to vectors and layers)."""
        ...

    @abstractmethod
    def relink_file(self, content_hash: str, new_file_path: str) -> bool:
        """Update file_path for a file matched by content_hash."""
        ...

    # ── Tier / Embedding info ───────────────────────────────────

    @abstractmethod
    def get_db_tier(self) -> Optional[str]:
        """Get the tier of existing data in the database."""
        ...

    @abstractmethod
    def get_db_embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension used in the database."""
        ...

    # ── FTS ──────────────────────────────────────────────────────

    @abstractmethod
    def rebuild_fts(self):
        """Rebuild full-text search index for all files."""
        ...

    # ── Lifecycle ────────────────────────────────────────────────

    @abstractmethod
    def close(self):
        """Close database connection."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
