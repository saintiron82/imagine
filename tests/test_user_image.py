import time
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

def test_image_vectorization():
    # User uploaded image path
    image_path = r"C:/Users/saint/.gemini/antigravity/brain/459a4ddb-4347-46e7-8541-00ff65acbde8/uploaded_media_1770278187213.jpg"
    
    print(f"1. Loading Image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        return

    img = Image.open(image_path)
    print(f"   - Original Size: {img.size}")
    
    print("\n2. Loading AI Model (clip-ViT-L-14 - High Performance)...")
    start_load = time.time()
    # GPU 사용 가능 시 자동으로 GPU 활용
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device.upper()}")
    model = SentenceTransformer('clip-ViT-L-14', device=device)
    load_time = time.time() - start_load
    print(f"   - Model Loaded in {load_time:.2f} seconds")

    print("\n3. Processing & Vectorizing...")
    start_infer = time.time()
    
    # Encode image
    vector = model.encode(img)
    
    infer_time = time.time() - start_infer
    print(f"   - Vectorization Time: {infer_time:.4f} seconds")
    print(f"   - Vector Dimension: {len(vector)}")
    print(f"   - Vector Sample (first 5 values): {vector[:5]}")

    print("\n4. Token Consumption Report")
    print("="*30)
    print("   [API Calls]: 0 (Running locally)")
    print("   [Tokens Used]: 0 (No token concept in raw inference)")
    print("   [Cost]: $0.00")
    print("="*30)
    
    # Optional: Zero-shot classification test
    print("\n5. AI Contents Guessing (Zero-shot)")
    candidates = ["night city", "sunny beach", "forest", "mountain", "cat"]
    print(f"   - Candidates: {candidates}")
    
    # Encode text and compare
    text_emb = model.encode(candidates)
    from sentence_transformers import util
    scores = util.cos_sim(vector, text_emb)[0]
    
    best_score_idx = scores.argmax()
    print(f"   - AI thinks this is: '{candidates[best_score_idx]}' (Confidence: {scores[best_score_idx]:.4f})")

if __name__ == "__main__":
    test_image_vectorization()
