"""
PSD Parser Module - Parses PSD files into AssetMeta format with full layer extraction.

This is the core parser for Photoshop files, handling:
- Layer hierarchy traversal
- Text layer content extraction
- Font information extraction
- Composite image rendering
- Thumbnail generation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Layer, Group

from .base_parser import BaseParser
from .schema import AssetMeta, ParseResult, LayerInfo
from .cleaner import clean_layer_name, infer_content_type, build_semantic_tags

logger = logging.getLogger(__name__)


class PSDParser(BaseParser):
    """
    Parser for Adobe Photoshop PSD files.
    
    Extracts full layer hierarchy, text content, fonts,
    and generates thumbnails for VV generation.
    """
    
    supported_extensions = ['psd']
    
    # THUMBNAIL_SIZE now resolved dynamically from tier config via get_thumbnail_max_edge()
    
    def parse(self, file_path: Path) -> ParseResult:
        """
        Parse a PSD file into AssetMeta format.
        
        Args:
            file_path: Path to the PSD file
            
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
            # Open PSD file
            psd = PSDImage.open(file_path)
            
            width, height = psd.width, psd.height
            
            # Extract layer information
            layer_tree, layer_paths, text_contents, fonts = self._extract_layers(
                psd, (width, height)
            )

            # Generate composite thumbnail
            thumbnail_path = self._create_thumbnail(psd, file_path)

            # Build semantic tags from layer paths
            # v3.1 Context Injection: Use raw layer names for richer context
            semantic_tags = self._build_semantic_tags_raw(psd)
            
            # Get file stats
            file_stats = file_path.stat()
            
            # Count layers
            layer_count = len(list(psd.descendants()))
            
            # Build AssetMeta
            asset_meta = AssetMeta(
                file_path=str(file_path.absolute()),
                file_name=file_path.name,
                file_size=file_stats.st_size,
                format='PSD',
                resolution=(width, height),
                visual_source_path=str(thumbnail_path) if thumbnail_path else None,
                semantic_tags=semantic_tags,
                text_content=text_contents,
                layer_tree=layer_tree,
                layer_count=layer_count,
                used_fonts=fonts,
                metadata={
                    'color_mode': str(psd.color_mode),
                    'channels': psd.channels,
                },
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
            logger.exception(f"Failed to parse PSD: {file_path}")
            errors.append(f"Failed to parse {file_path}: {str(e)}")
            return ParseResult(success=False, errors=errors)
    
    def _extract_layers(
        self, 
        psd: PSDImage, 
        canvas_size: tuple[int, int]
    ) -> tuple[Dict, List[str], List[str], List[str]]:
        """
        Extract layer hierarchy, text, and fonts from PSD.
        
        Returns:
            Tuple of (layer_tree, layer_paths, text_contents, fonts)
        """
        layer_paths = []
        text_contents = []
        fonts = set()
        
        def process_layer(layer, parent_path: str = "") -> Dict[str, Any]:
            """Recursively process a layer and its children."""
            layer_name = layer.name or "Unnamed"
            cleaned_name = clean_layer_name(layer_name)
            
            # Build path
            current_path = f"{parent_path}/{layer_name}" if parent_path else layer_name
            if cleaned_name:
                layer_paths.append(current_path)
            
            # Get layer kind
            layer_kind = getattr(layer, 'kind', 'unknown')
            if layer_kind is None:
                layer_kind = 'unknown'
            
            # Infer content type
            content_type = infer_content_type(
                layer_name=layer_name,
                canvas_size=canvas_size,
                layer_size=(layer.width, layer.height),
                layer_position=(layer.left, layer.top),
            )
            
            layer_info = {
                "name": layer_name,
                "cleaned_name": cleaned_name,
                "kind": str(layer_kind),
                "visible": layer.visible,
                "opacity": layer.opacity / 255.0 if hasattr(layer, 'opacity') else 1.0,
                "position": [layer.left, layer.top],
                "size": [layer.width, layer.height],
                "content_type": content_type,
            }
            
            # Extract text content
            if str(layer_kind) == 'type':
                text_info = self._extract_text_layer(layer)
                if text_info.get('text'):
                    text_contents.append(text_info['text'])
                    layer_info['text'] = text_info['text']
                if text_info.get('fonts'):
                    fonts.update(text_info['fonts'])
            
            # Process children for groups
            if str(layer_kind) == 'group':
                children = []
                for child in layer:
                    children.append(process_layer(child, current_path))
                if children:
                    layer_info['children'] = children
            
            return layer_info
        
        # Build root tree
        root = {
            "name": "Root",
            "size": list(canvas_size),
            "children": []
        }
        
        for layer in psd:
            root['children'].append(process_layer(layer))
        
        return root, layer_paths, text_contents, list(fonts)
    
    def _extract_text_layer(self, layer) -> Dict[str, Any]:
        """Extract text content and fonts from a text layer."""
        result = {'text': None, 'fonts': []}
        
        try:
            # Try to get text from engine_dict
            if hasattr(layer, 'engine_dict') and layer.engine_dict:
                engine_data = layer.engine_dict
                
                # Try to extract text
                if 'Editor' in engine_data and 'Text' in engine_data['Editor']:
                    text_val = engine_data['Editor']['Text']
                    result['text'] = (text_val.value if hasattr(text_val, 'value') else str(text_val)).rstrip('\r\n')
                
                # Try to extract fonts from ResourceDict
                if 'ResourceDict' in engine_data:
                    resource = engine_data['ResourceDict']
                    if 'FontSet' in resource:
                        for font_info in resource['FontSet']:
                            if 'Name' in font_info:
                                result['fonts'].append(font_info['Name'])
            
            # Fallback: try text property
            if not result['text'] and hasattr(layer, 'text'):
                text_val = layer.text
                result['text'] = (text_val.value if hasattr(text_val, 'value') else str(text_val)).rstrip('\r\n')
                
        except Exception as e:
            logger.warning(f"Failed to extract text from layer {layer.name}: {e}")
        
        return result
    
    def _create_thumbnail(self, psd: PSDImage, file_path: Path) -> Optional[Path]:
        """Create a thumbnail from the PSD composite image."""
        try:
            thumbnail_path = self.get_thumbnail_path(file_path)
            
            # Get composite image
            composite = psd.composite()
            
            if composite is None:
                logger.warning(f"No composite image available for {file_path}")
                return None
            
            # Convert to RGB if necessary
            if composite.mode in ('RGBA', 'P'):
                background = Image.new('RGB', composite.size, (255, 255, 255))
                if composite.mode == 'P':
                    composite = composite.convert('RGBA')
                background.paste(composite, mask=composite.split()[-1] if composite.mode == 'RGBA' else None)
                composite = background
            elif composite.mode != 'RGB':
                composite = composite.convert('RGB')
            
            # Resize maintaining aspect ratio (tier-aware max edge)
            max_edge = self.get_thumbnail_max_edge()
            composite.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            composite.save(thumbnail_path, 'PNG')
            return thumbnail_path
            
        except Exception as e:
            logger.warning(f"Failed to create thumbnail for {file_path}: {e}")
            return None
    
    def _save_json(self, asset_meta: AssetMeta, file_path: Path) -> None:
        """Save AssetMeta to JSON file."""
        json_path = self.get_json_path(file_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(asset_meta.model_dump_json(indent=2))

    def _build_semantic_tags_raw(self, psd: PSDImage) -> str:
        """
        Build semantic tags from raw layer names for Context Injection.

        Unlike build_semantic_tags() which uses clean_layer_name(),
        this method preserves original layer names for richer AI context.

        Args:
            psd: PSDImage object

        Returns:
            Space-separated layer names (max 50 layers)
        """
        from .cleaner import MEANINGLESS_NAMES

        layer_names = []

        def collect_layer_names(layer, depth=0):
            """Recursively collect layer names."""
            if len(layer_names) >= 50:  # Limit to 50 layers
                return

            layer_name = layer.name or ""
            if layer_name:
                # Skip completely meaningless names
                name_lower = layer_name.lower().strip()
                if name_lower not in MEANINGLESS_NAMES and layer_name.strip():
                    # Keep original name but normalize spacing
                    normalized = ' '.join(layer_name.split())
                    if normalized and normalized not in layer_names:
                        layer_names.append(normalized)

            # Process children for groups
            if hasattr(layer, '__iter__'):
                for child in layer:
                    collect_layer_names(child, depth + 1)

        # Collect from all top-level layers
        for layer in psd:
            collect_layer_names(layer)

        return ' '.join(layer_names[:50])
