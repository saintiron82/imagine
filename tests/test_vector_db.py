import chromadb
import uuid
from typing import List

def test_chromadb_setup():
    print("Initialize ChromaDB Client...")
    # Using an ephemeral client for testing (in-memory)
    # For persistence, we would use: chromadb.PersistentClient(path="./chroma_db")
    client = chromadb.Client()
    
    collection_name = "image_metadata_test"
    
    # Clean up if exists
    try:
        client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Collection cleanup skipped (not found or error): {e}")
        
    print(f"Creating collection '{collection_name}'...")
    collection = client.create_collection(name=collection_name)
    
    # Sample data: Simulating image descriptions
    documents = [
        "A photo of a cute cat on a sofa",
        "A screenshot of a coding interface with dark mode",
        "A landscape with mountains and a lake",
        "An error message showing a stack trace",
        "A diagram explaining software architecture"
    ]
    
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    
    # Metadatas (optional)
    metadatas = [{"source": "test", "type": "image"} for _ in range(len(documents))]
    
    print("Adding documents to collection...")
    # Chroma uses a default embedding model (all-MiniLM-L6-v2) if none is specified
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Verifying count...")
    count = collection.count()
    print(f"Total documents: {count}")
    assert count == 5
    
    # Test Search
    query_text = "programming code screen"
    print(f"\nQuerying for: '{query_text}'")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    
    print("\nResults:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i+1}. {doc} (Distance: {results['distances'][0][i]:.4f})")
        
    # Validation: We expect the coding interface or error message to be returned
    expected_keywords = ["coding", "stack trace", "software"]
    found_relevant = any(any(k in r for k in expected_keywords) for r in results['documents'][0])
    
    if found_relevant:
        print("\n[SUCCESS] Relevant documents found!")
    else:
        print("\n[WARNING] Results might not be relevant. Check embeddings.")

if __name__ == "__main__":
    try:
        test_chromadb_setup()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        exit(1)
