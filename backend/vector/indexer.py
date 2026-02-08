"""
Vector Indexer - Handles embedding generation and ChromaDB ingestion.
"""
import logging
import gc
import torch
import chromadb
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from PIL import Image

logger = logging.getLogger(__name__)

class VectorIndexer:
    def __init__(self, db_path: Optional[Path] = None, device: str = None):
        if db_path is None:
            # Default to project root / chroma_db
            db_path = Path(__file__).parent.parent.parent / "chroma_db"
            
        self.db_path = db_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DB
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.image_collection = self.client.get_or_create_collection(name="image_library")
            self.text_collection = self.client.get_or_create_collection(name="text_library") # Future use
            logger.info(f"VectorDB Ready at {self.db_path} (Images: {self.image_collection.count()})")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

        # Load AI Model (Lazy Load)
        self.model = None

    def _load_model(self):
        if not self.model:
            logger.info(f"Loading CLIP Model (ViT-L-14) on {self.device.upper()}...")
            try:
                self.model = SentenceTransformer('clip-ViT-L-14', device=self.device)
            except Exception as e:
                logger.error(f"Failed to load AI model: {e}")
                raise

    def index_image(self, file_path: Path, metadata: Dict[str, Any], thumbnail_path: Optional[Path] = None):
        """
        Generates visual embedding for the image and saves to ChromaDB.
        Uses thumbnail if provided, otherwise reads original file.
        """
        if not self.client:
            return
            
        try:
            self._load_model()
            
            # Prefer thumbnail for speed, fallback to original
            target_path = thumbnail_path if thumbnail_path and thumbnail_path.exists() else file_path
            
            # Load & Preprocess
            image = Image.open(target_path).convert("RGB")
            # Resize if reading original (CLIP default is 224, but keeping some detail for resize)
            if target_path == file_path:
                image.thumbnail((512, 512)) # Reasonable limit
                
            # Embed
            embedding = self.model.encode(image)
            
            # Prepare Metadata
            # ChromaDB requires flat metadata (str, int, float, bool)
            safe_metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "format": metadata.get('format', ''),
                "width": metadata.get('resolution', (0,0))[0],
                "height": metadata.get('resolution', (0,0))[1],
                "tags": metadata.get('translated_tags', metadata.get('semantic_tags', ''))[:1000] if metadata.get('semantic_tags') else '',
                "ai_caption": metadata.get('ai_caption', '')[:500] if metadata.get('ai_caption') else '',
                "dominant_color": metadata.get('dominant_color', ''),
            }

            # Build rich document for keyword search (Phase 4)
            # Combine all text sources for hybrid search
            doc_parts = [file_path.name]

            # Add AI-generated caption
            if metadata.get('ai_caption'):
                doc_parts.append(f"Caption: {metadata['ai_caption']}")

            # Add AI tags
            if metadata.get('ai_tags'):
                doc_parts.append(f"AI Tags: {', '.join(metadata['ai_tags'])}")

            # Add semantic tags
            if metadata.get('semantic_tags'):
                doc_parts.append(f"Tags: {metadata['semantic_tags']}")

            # Add OCR text
            if metadata.get('ocr_text'):
                doc_parts.append(f"Text: {metadata['ocr_text']}")

            # Add layer text content
            if metadata.get('text_content'):
                doc_parts.append(f"Content: {' '.join(metadata['text_content'])}")

            # Join all parts
            document = " | ".join(doc_parts)[:2000]  # Limit document size

            # Upsert
            doc_id = str(file_path).replace("\\", "/")  # Normalize ID
            self.image_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                metadatas=[safe_metadata],
                documents=[document]
            )
            
            logger.info(f"Indexed: {file_path.name}")
            
            # Cleanup
            del image, embedding
            # Periodic cleanup
            if self.device == 'cuda' and self.image_collection.count() % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Indexing Failed for {file_path.name}: {e}")

    def index_text(self, file_path: Path, text: str):
        """Future: Add text-only embedding for specific keyword search"""
        pass
