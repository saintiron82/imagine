"""
Base Parser Module - Abstract base class for all image parsers.

All concrete parsers (PSD, PNG, JPG) must inherit from BaseParser
and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .schema import AssetMeta, ParseResult


class BaseParser(ABC):
    """
    Abstract base class for image asset parsers.
    
    Concrete implementations must provide:
    - parse(): Main parsing logic
    - supported_extensions: List of file extensions this parser handles
    """
    
    supported_extensions: list[str] = []
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the parser.
        
        Args:
            output_dir: Directory to save generated assets (thumbnails, etc.)
        """
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        """
        Parse an image file and extract metadata.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ParseResult containing AssetMeta or errors
        """
        pass
    
    @classmethod
    def can_parse(cls, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this parser supports the file extension
        """
        return file_path.suffix.lower().lstrip('.') in [
            ext.lower().lstrip('.') for ext in cls.supported_extensions
        ]
    
    def validate_file(self, file_path: Path) -> tuple[bool, str]:
        """
        Validate that the file exists and is readable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        if not file_path.is_file():
            return False, f"Not a file: {file_path}"
        
        if not self.can_parse(file_path):
            return False, f"Unsupported extension: {file_path.suffix}"
        
        return True, ""
    
    @staticmethod
    def get_thumbnail_max_edge() -> int:
        """Get tier-aware thumbnail max edge from config."""
        try:
            from backend.utils.tier_config import get_active_tier
            _, tier_config = get_active_tier()
            return tier_config.get("preprocess", {}).get("max_edge", 768)
        except Exception:
            return 768  # safe default for pro

    def get_thumbnail_path(self, file_path: Path) -> Path:
        """
        Generate the path for the thumbnail file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Path where thumbnail should be saved
        """
        thumbnail_dir = self.output_dir / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        return thumbnail_dir / f"{file_path.stem}_thumb.png"
    
    def get_json_path(self, file_path: Path) -> Path:
        """
        Generate the path for the JSON metadata file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Path where JSON should be saved
        """
        json_dir = self.output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        return json_dir / f"{file_path.stem}.json"
