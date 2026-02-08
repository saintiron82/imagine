# AI & Vector System Architecture

## 1. AI Engine (The Eye)
We use a **Local Multimodal AI** that runs entirely on your machine.

- **Model**: `CLIP (ViT-L-14)` by OpenAI (via `sentence-transformers`)
    - **Performance**: High Accuracy (Legacy model was B-32)
    - **Device**: **GPU (NVIDIA GeForce RTX 3060 Ti)**
    - **VRAM Usage**: Approx 1GB
- **Capabilities**:
    - **Vision**: Converts images into 768-dimensional vectors.
    - **Zero-shot**: Can match images to text without explicit training.
    - **Multilingual**: Supports English native, extensible to Korean via translation.

## 2. Vector Database (The Memory)
We use **ChromaDB** as the high-performance local vector store.

- **Type**: Embedded Vector Database (No server installation required)
- **Location**: `c:\Users\saint\ImageParser\chroma_db` (Persistent storage)
- **Schema**:
    - **ID**: File Path (Unique Identifier)
    - **Embedding**: 768 float values (The "meaning" of the image)
    - **Metadata**: Filename, Folder, File Size, Width, Height
    - **Document**: Raw filename (for keyword basics)

## 3. Workflow (The Pipeline)
How an image is processed:

1.  **Input**: Image File (JPG, PNG, PSD-Composite)
2.  **Preprocessing**: Resize to 224x224 (Standard CLIP input)
3.  **Inference (AI)**:
    - Image passes through ViT-L-14 Model on GPU.
    - Output: `[0.12, -0.59, 0.99, ...]` (768 numbers)
4.  **Storage**: Vector + Metadata saved to ChromaDB.
5.  **Search**:
    - User types "Water Reflection".
    - Text converted to Vector.
    - DB finds nearest Image Vectors.

## 4. Current Status
- **Installed Components**:
    - `sentence-transformers` (AI Loader)
    - `torch` (GPU Backend, CUDA 12.1)
    - `chromadb` (Database)
- **Performance Benchmark**:
    - **Speed**: ~0.15s per image
    - **Cost**: $0 (Local Execution)
