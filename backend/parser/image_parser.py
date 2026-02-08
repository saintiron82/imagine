"""
Image Parser Module - Parses PNG and JPG files into AssetMeta format.

For non-PSD images, we treat them as single-layer structures
and extract available metadata (EXIF, dimensions, etc.).
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
import exifread

from .base_parser import BaseParser
from .schema import AssetMeta, ParseResult, LayerInfo
from .cleaner import clean_layer_name, infer_content_type


class ImageParser(BaseParser):
    """
    Parser for standard image formats (PNG, JPG, JPEG).
    
    Creates a normalized AssetMeta structure treating the image
    as a single-layer document.
    """
    
    supported_extensions = ['png', 'jpg', 'jpeg']
    
    THUMBNAIL_SIZE = (512, 512)  # For vision AI and UI display
    
    def parse(self, file_path: Path) -> ParseResult:
        """
        Parse a PNG or JPG file into AssetMeta format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ParseResult with AssetMeta or errors
        """
        errors = []
        warnings = []
        
        # Validate file
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            return ParseResult(success=False, errors=[error_msg])
        
        try:
            # Open image with PIL
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Determine format
                img_format = file_path.suffix.upper().lstrip('.')
                if img_format == 'JPEG':
                    img_format = 'JPG'
                
                # Generate thumbnail
                thumbnail_path = self._create_thumbnail(img, file_path)
                
                # Extract EXIF metadata (for JPG)
                exif_data = self._extract_exif(file_path)
                
                # Build layer tree (single virtual layer for images)
                layer_name = clean_layer_name(file_path.stem)
                if not layer_name:
                    layer_name = "Image"
                
                # Infer content type
                content_type = infer_content_type(
                    layer_name=layer_name,
                    canvas_size=(width, height),
                    layer_size=(width, height),
                    layer_position=(0, 0),
                    alpha_ratio=1.0
                )
                
                layer_tree = {
                    "name": "Root",
                    "children": [
                        {
                            "name": layer_name,
                            "type": content_type,
                            "size": [width, height],
                            "position": [0, 0]
                        }
                    ]
                }
                
                # Build semantic tags from filename
                semantic_tags = layer_name if layer_name else file_path.stem
                
                # Get file stats
                file_stats = file_path.stat()
                
                # Build AssetMeta
                asset_meta = AssetMeta(
                    file_path=str(file_path.absolute()),
                    file_name=file_path.name,
                    file_size=file_stats.st_size,
                    format=img_format,
                    resolution=(width, height),
                    visual_source_path=str(thumbnail_path) if thumbnail_path else None,
                    semantic_tags=semantic_tags,
                    text_content=[],  # No text in regular images (OCR could be added later)
                    layer_tree=layer_tree,
                    layer_count=1,
                    used_fonts=[],
                    metadata=exif_data,
                    modified_at=datetime.fromtimestamp(file_stats.st_mtime),
                    thumbnail_url=str(thumbnail_path) if thumbnail_path else None
                )
                
                # Save JSON
                self._save_json(asset_meta, file_path)
                
                return ParseResult(
                    success=True,
                    asset_meta=asset_meta,
                    warnings=warnings
                )
                
        except Exception as e:
            errors.append(f"Failed to parse {file_path}: {str(e)}")
            return ParseResult(success=False, errors=errors)
    
    def _create_thumbnail(self, img: Image.Image, file_path: Path) -> Optional[Path]:
        """Create a thumbnail for visual embedding."""
        try:
            thumbnail_path = self.get_thumbnail_path(file_path)
            
            # Convert to RGB if necessary (for CLIP compatibility)
            if img.mode in ('RGBA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize maintaining aspect ratio, then center crop
            img.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            img.save(thumbnail_path, 'PNG')
            return thumbnail_path
            
        except Exception as e:
            print(f"Warning: Failed to create thumbnail: {e}")
            return None
    
    def _extract_exif(self, file_path: Path) -> dict:
        """Extract EXIF metadata from the image."""
        exif_data = {}
        
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                # Extract useful tags
                useful_tags = [
                    'EXIF DateTimeOriginal',
                    'Image Artist',
                    'Image Copyright',
                    'Image Software',
                    'EXIF ExifImageWidth',
                    'EXIF ExifImageLength',
                ]
                
                for tag in useful_tags:
                    if tag in tags:
                        key = tag.split()[-1].lower()
                        exif_data[key] = str(tags[tag])
                        
        except Exception:
            pass  # EXIF extraction is optional
        
        return exif_data
    
    def _save_json(self, asset_meta: AssetMeta, file_path: Path) -> None:
        """Save AssetMeta to JSON file."""
        json_path = self.get_json_path(file_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(asset_meta.model_dump_json(indent=2))
