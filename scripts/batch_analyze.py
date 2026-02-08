"""
Batch Folder Image Analyzer (Memory-Safe Version)
- Processes ONE image at a time
- Clears GPU memory after each image
- Uses CPU if GPU causes issues
"""
import os
import time
import gc
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

# Configuration
TARGET_FOLDER = r"C:\Images\밤"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
USE_GPU = True  # 하나씩 처리하므로 GPU 사용 가능

def main():
    print("=" * 50)
    print("🖼️  Batch Image Analyzer (Safe Mode)")
    print("=" * 50)
    
    # 1. Check folder
    folder = Path(TARGET_FOLDER)
    if not folder.exists():
        print(f"[ERROR] Folder not found: {TARGET_FOLDER}")
        return
    
    # 2. Find images
    print(f"\n📂 Scanning: {TARGET_FOLDER}")
    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"   Found {len(image_files)} image(s)")
    
    if not image_files:
        print("[WARNING] No images found in folder.")
        return
    
    # 3. Load AI Model (CPU for stability)
    device = 'cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
    print(f"\n🧠 Loading AI Model (ViT-L-14 on {device.upper()})...")
    model = SentenceTransformer('clip-ViT-L-14', device=device)
    print("   Model loaded!")
    
    # 4. Initialize ChromaDB
    print("\n🗄️  Initializing ChromaDB...")
    db_path = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="image_library")
    print(f"   Collection ready (existing: {collection.count()} items)")
    
    # 5. Process images ONE BY ONE
    print("\n🔄 Processing images (one at a time)...")
    start_time = time.time()
    success_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"   [{i}/{len(image_files)}] {img_path.name}...", end=" ", flush=True)
            
            # Load image
            img = Image.open(img_path).convert("RGB")
            
            # Resize to reduce memory (CLIP uses 224x224 anyway)
            img.thumbnail((512, 512))
            
            # Vectorize
            vector = model.encode(img)
            
            # Metadata
            metadata = {
                "file_name": img_path.name,
                "file_path": str(img_path),
                "folder": str(img_path.parent),
            }
            
            # Save to DB
            doc_id = str(img_path).replace("\\", "/")
            collection.upsert(
                ids=[doc_id],
                embeddings=[vector.tolist()],
                metadatas=[metadata],
                documents=[img_path.name]
            )
            
            print("✅")
            success_count += 1
            
            # Memory cleanup
            del img, vector
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {e}")
    
    elapsed = time.time() - start_time
    
    # 6. Summary
    print("\n" + "=" * 50)
    print("📊 Complete!")
    print("=" * 50)
    print(f"   Success: {success_count}/{len(image_files)}")
    print(f"   Time: {elapsed:.1f}s ({elapsed/len(image_files):.2f}s/image)")
    print(f"   DB Total: {collection.count()} items")

if __name__ == "__main__":
    main()
    print("\n💡 You can now search these images semantically!")

if __name__ == "__main__":
    main()
