import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os

# Target Image (The one user said is wrong)
TARGET_IMAGE = r"C:\Images\밤\Ext1038.jpg"

# Diverse categories to probe what the AI thinks this is
CANDIDATES = [
    "water reflection", "lake", "ocean", "river",  # Water related
    "neon lights", "night street", "city building", "skyscraper", # City related
    "person", "portrait", "face", "crowd", # Human
    "food", "plate", "restaurant", # Object
    "text", "screenshot", "map", "document", # Abstract
    "car", "traffic", "train", "airplane", # Vehicle
    "indoors", "room", "bedroom", "office", # Indoor
    "mountain", "tree", "flower", "sky" # Nature
]

def diagnose():
    if not os.path.exists(TARGET_IMAGE):
        print(f"File not found: {TARGET_IMAGE}")
        return

    print(f"🕵️ Diagnosing: {TARGET_IMAGE}")
    
    # Load Model (GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Loading ViT-L-14 on {device.upper()}...")
    model = SentenceTransformer('clip-ViT-L-14', device=device)
    
    # Load Image
    img = Image.open(TARGET_IMAGE)
    print(f"   Image Size: {img.size}")
    
    # Encode Image
    img_emb = model.encode(img)
    
    # Encode Text Candidates
    text_emb = model.encode(CANDIDATES)
    
    # Calculate Similarities
    scores = util.cos_sim(img_emb, text_emb)[0]
    
    # Sort results
    results = sorted(zip(CANDIDATES, scores), key=lambda x: x[1], reverse=True)
    
    print("\n🧐 AI's Top 10 Guesses:")
    print("-" * 30)
    for label, score in results[:10]:
        print(f"   {label:<15} : {score:.4f}")
        
    print("\n📉 Checking 'water reflection' rank:")
    for i, (label, score) in enumerate(results):
        if label == "water reflection":
            print(f"   Rank #{i+1} : {score:.4f}")
            break

if __name__ == "__main__":
    diagnose()
