"""
Semantic Search - Test searching images by text query
"""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import torch

def search_images(query: str, top_k: int = 5):
    print(f"\n🔍 Searching: '{query}'")
    print("-" * 40)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('clip-ViT-L-14', device=device)
    
    # Encode query text
    query_vector = model.encode(query)
    
    # Connect to DB
    db_path = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name="image_library")
    
    # Search
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k
    )
    
    # Display results
    print(f"\n📋 Top {top_k} Results:")
    for i, (doc_id, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    ), 1):
        similarity = 1 - distance  # Convert distance to similarity
        print(f"   {i}. {metadata['file_name']}")
        print(f"      📁 {metadata['folder']}")
        print(f"      📊 Similarity: {similarity:.4f}")
        print()

if __name__ == "__main__":
    # Test search
    search_images("물 반사")  # Water reflection
