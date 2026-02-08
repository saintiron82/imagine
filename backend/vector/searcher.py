"""
Vector Searcher - Handles semantic search queries.
"""
import logging
import torch
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorSearcher:
    def __init__(self, db_path: Optional[Path] = None, device: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "chroma_db"
            
        self.db_path = db_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.image_collection = self.client.get_collection(name="image_library")
        except Exception as e:
            logger.error(f"Searcher Init Failed: {e}")
            self.client = None

        self.model = None

    def _load_model(self):
        if not self.model:
            logger.info("Loading CLIP for Search...")
            try:
                self.model = SentenceTransformer('clip-ViT-L-14', device=self.device)
            except Exception as e:
                logger.error(f"Model Load Failed: {e}")
                raise

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for images matching the text query.
        """
        if not self.client or not self.image_collection:
            return []

        try:
            self._load_model()
            
            # Encode query
            query_vector = self.model.encode(query)
            
            # Query DB
            results = self.image_collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results and results['ids']:
                ids = results['ids'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc_id, meta, dist in zip(ids, metadatas, distances):
                    score = 1 - dist # Rough conversion
                    formatted_results.append({
                        "id": doc_id,
                        "file_path": meta.get('file_path'),
                        "file_name": meta.get('file_name'),
                        "score": float(score),
                        "metadata": meta
                    })
            
            return formatted_results

        except Exception as e:
            logger.error(f"Search Failed: {e}")
            return []
