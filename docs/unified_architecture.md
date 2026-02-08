# 통합 아키텍처: PSD 레이어 단위 지능형 분석 시스템

## 🎯 비전 통합

### 사용자의 핵심 비전
> **"Qwen은 이미지의 '내용물'을 텍스트로 적어두는 서기이고, CLIP은 이미지의 '인상'을 기억하는 목격자입니다."**

이 비전을 ImageParser 프로젝트에 통합하여 **파일 단위 → 레이어 단위** 분석으로 확장합니다.

---

## 🏗️ 최종 시스템 아키텍처

### 3-Axis + 레이어 분해 통합 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    PSD 파일 (입력)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Structural Parsing (구조적 분석)                   │
│  - psd-tools로 레이어 트리 분해                               │
│  - 각 레이어를 PNG로 렌더링                                   │
│  - 좌표, 크기, 타입, 폰트 추출                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────────┐            ┌──────────────────────┐
│  Phase 2: Latent     │            │  Phase 4: Descriptive│
│  (감각적 검색)        │            │  (지능형 분석)        │
│                      │            │                      │
│  OpenCLIP ViT-L/14   │            │  Vision LM           │
│  - 768차원 벡터       │            │  (Qwen or Florence)  │
│  - 시각적 유사도      │            │  - OCR 텍스트        │
│  - "무드" 검색        │            │  - 상세 캡션         │
│  - 0.01초/query      │            │  - 태그 추출         │
│                      │            │  - 좌표 분석         │
└──────────────────────┘            └──────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: ChromaDB Storage (통합 저장소)                     │
│  - 레이어 단위로 인덱싱                                       │
│  - 벡터 + 메타데이터 통합                                     │
│  - 하이브리드 검색 (Vector + Keyword + Filter)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 데이터 스키마 (사용자 비전 반영)

### ChromaDB 레이어 레코드 구조

```python
{
    # === 고유 식별자 ===
    "id": "proj_summer_2026_001_layer_052",

    # === 검색 대상 문서 (Qwen 캡션) ===
    "document": "해변 배경에 세일 문구가 적힌 프로모션 배너",

    # === 임베딩 벡터 (CLIP 자동 생성) ===
    "embedding": [0.12, -0.59, 0.99, ...],  # 768차원

    # === 메타데이터 (검색 필터링용) ===
    "metadata": {
        # 1. 출처 정보
        "source_file": "C:/Assets/summer_promo.psd",
        "layer_index": 52,
        "layer_name": "Banner_50%_Sale",
        "layer_path": "Group1/Promo/Banner_50%_Sale",

        # 2. 물리/구조 정보 (Python 추출)
        "layer_type": "pixel",  # text, pixel, shape, group
        "width": 1200,
        "height": 400,
        "pos_x": 150,
        "pos_y": 300,
        "canvas_width": 1920,
        "canvas_height": 1080,

        # 3. 비전 분석 정보 (Qwen/Florence 추출)
        "ai_caption": "해변 배경에 세일 문구가 적힌 프로모션 배너",
        "ai_tags": "summer,beach,banner,sale,promotion",
        "ai_style": "minimalist,modern,bright",
        "dominant_color": "#007BFF",
        "color_palette": "#007BFF,#FFFFFF,#FFD700",

        # 4. 텍스트/OCR (Qwen/Python 추출)
        "ocr_text": "SUMMER BIG SALE 50%",
        "font_family": "Helvetica Neue Bold",
        "text_color": "#FFFFFF",

        # 5. 관리 정보
        "user_tags": "",  # 사용자 커스텀 태그
        "project_name": "summer_campaign_2026",
        "status": "untagged",  # untagged, approved, pending, archived
        "created_at": "2026-02-06T13:30:00Z",
        "modified_at": "2026-02-06T13:30:00Z",

        # 6. 파일 단위 정보 (기존 Phase 1-3)
        "file_format": "PSD",
        "file_size_mb": 45.2,
        "layer_count_total": 127,
    }
}
```

---

## 🔍 검색 시나리오 (하이브리드 쿼리)

### 1. 자연어 검색 (CLIP 벡터)
```python
query = "시원한 느낌의 파란색 버튼"
results = collection.query(
    query_texts=[query],
    n_results=20
)
# CLIP이 시각적 유사도로 검색
```

### 2. 정밀 필터링 (메타데이터)
```python
results = collection.query(
    query_texts=["버튼"],
    where={
        "$and": [
            {"ocr_text": {"$contains": "50%"}},
            {"width": {"$gte": 500}},
            {"layer_type": {"$eq": "pixel"}},
            {"dominant_color": {"$contains": "blue"}}
        ]
    },
    n_results=10
)
```

### 3. 복합 검색 (AI + 사용자 태그)
```python
results = collection.query(
    query_texts=["고양이"],
    where={
        "$and": [
            {"ai_tags": {"$contains": "cat"}},
            {"user_tags": {"$contains": "A프로젝트"}},
            {"status": {"$eq": "approved"}}
        ]
    }
)
```

### 4. 위치 기반 검색
```python
# "오른쪽 하단 배너만 찾기"
results = collection.query(
    query_texts=["배너"],
    where={
        "$and": [
            {"pos_x": {"$gte": 1200}},  # 오른쪽
            {"pos_y": {"$gte": 700}},   # 하단
            {"layer_type": {"$ne": "group"}}  # 그룹 제외
        ]
    }
)
```

---

## 🖥️ 환경별 최적화 전략

### Windows (RTX 3060 Ti) - 현재 프로젝트
| 컴포넌트 | 모델 | VRAM | 속도 |
|---------|------|------|------|
| **Visual Embedding** | CLIP ViT-L-14 | 2GB | 0.5초/image |
| **Vision Analysis** | Florence-2-large | 2GB | 0.5초/image |
| **Total** | - | ~4GB | ~1초/layer |

**장점:**
- Florence-2는 Microsoft 공식, 안정적
- Object Detection, OCR 통합 지원
- Windows 생태계와 호환성 우수

### MacBook M5 (32GB) - 사용자 환경
| 컴포넌트 | 모델 | 메모리 | 속도 |
|---------|------|--------|------|
| **Visual Embedding** | CLIP ViT-L-14 | 2GB | 0.3초/image (NPU) |
| **Vision Analysis** | Qwen2.5-VL (7B) | 8GB | 0.8초/image (MLX) |
| **Total** | - | ~10GB | ~1초/layer |

**장점:**
- Qwen은 다국어(중/영/한) 강력
- MLX/Ollama로 NPU 가속
- 통합 메모리 32GB로 배치 처리 최적화

**권장:**
- 배치 크기: 32개 (32GB ÷ 10GB ≈ 3배치)
- 병렬 처리: 4 workers
- 예상 처리량: 1000 레이어 → 30분~1시간

---

## 🔄 파이프라인 통합 (Phase 4 수정)

### 기존 계획 (파일 단위)
```python
# ingest_engine.py (기존)
def process_file(file_path):
    meta = parser.parse(file_path)  # 파일 전체 분석
    vision_result = analyzer.analyze(thumbnail)  # 썸네일 1장만
    meta.ai_caption = vision_result["caption"]
    indexer.index_image(file_path, meta)  # 파일 1개 인덱싱
```

### 확장 계획 (레이어 단위)
```python
# ingest_engine.py (확장)
def process_file(file_path):
    # 1. 파일 단위 분석 (기존)
    file_meta = parser.parse(file_path)

    # 2. 레이어 단위 분석 (NEW)
    layer_analyzer = LayerVisionAnalyzer()

    for layer in file_meta.layer_tree:
        # 각 레이어를 PNG로 렌더링
        layer_image = render_layer(psd, layer)

        # CLIP + Vision 분석
        layer_result = layer_analyzer.analyze(
            image=layer_image,
            context={
                "layer_name": layer.name,
                "layer_type": layer.kind,
                "position": (layer.left, layer.top),
                "size": (layer.width, layer.height)
            }
        )

        # ChromaDB에 레이어 단위로 인덱싱
        layer_record = {
            "id": f"{file_hash}_{layer.index}",
            "document": layer_result["caption"],
            "metadata": {
                "source_file": str(file_path),
                "layer_index": layer.index,
                "layer_name": layer.name,
                "layer_path": layer.path,
                "layer_type": layer.kind,
                "width": layer.width,
                "height": layer.height,
                "pos_x": layer.left,
                "pos_y": layer.top,
                # Vision 분석 결과
                "ai_caption": layer_result["caption"],
                "ai_tags": ",".join(layer_result["tags"]),
                "ai_style": layer_result.get("style", ""),
                "dominant_color": layer_result.get("color", ""),
                "ocr_text": layer_result.get("ocr", ""),
                # 관리 정보
                "user_tags": "",
                "status": "untagged",
                "created_at": datetime.now().isoformat()
            }
        }

        indexer.index_layer(layer_record)
```

---

## 📋 Phase 4 재정의 (레이어 단위 분석)

### U-015: 레이어 렌더링 및 기반 구축
**목표:**
- PSD 레이어를 개별 PNG로 렌더링
- LayerVisionAnalyzer 클래스 구현
- 레이어 컨텍스트 정보 추출

**구현:**
```python
# backend/vision/layer_renderer.py
def render_layer(psd, layer, output_size=(512, 512)):
    """레이어를 독립적인 PNG로 렌더링"""
    layer_image = layer.composite()
    if layer_image:
        layer_image.thumbnail(output_size)
        return layer_image
    return None
```

**테스트:**
```python
renderer = LayerRenderer()
image = renderer.render_layer(psd, layer)
assert image is not None
assert image.size[0] <= 512
```

---

### U-016: Vision 모델 통합 (환경별)

**Windows (Florence-2):**
```python
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
).to("cuda")

def analyze_layer(image, context):
    prompt = f"<DETAILED_CAPTION> This is a {context['layer_type']} layer"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    result = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(result[0])
```

**MacBook M5 (Qwen2.5-VL):**
```python
# Ollama API 사용
import ollama

def analyze_layer(image, context):
    prompt = f"Describe this {context['layer_type']} layer in detail"
    response = ollama.chat(
        model="qwen2.5-vl:7b",
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image]
        }]
    )
    return response['message']['content']
```

---

### U-017: ChromaDB 스키마 확장

**레이어 컬렉션 생성:**
```python
# backend/vector/indexer.py
class LayerIndexer:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.layer_collection = self.client.get_or_create_collection(
            name="layer_library",
            metadata={"description": "PSD layer-level analysis"}
        )

    def index_layer(self, layer_record: Dict[str, Any]):
        """레이어 단위 인덱싱"""
        self.layer_collection.upsert(
            ids=[layer_record["id"]],
            documents=[layer_record["document"]],
            metadatas=[layer_record["metadata"]],
            embeddings=[self._get_embedding(layer_record["image"])]
        )
```

---

### U-018: 하이브리드 검색 구현

```python
# backend/vector/searcher.py
class LayerSearcher:
    def search_hybrid(
        self,
        query: str,
        filters: Dict = None,
        top_k: int = 20
    ):
        """
        하이브리드 검색:
        1. CLIP 벡터 검색 (70%)
        2. 메타데이터 필터링 (정밀)
        3. 키워드 매칭 (30%)
        """
        # Vector 검색
        results = self.layer_collection.query(
            query_texts=[query],
            where=filters,
            n_results=top_k * 2
        )

        # 키워드 재스코어링
        query_words = set(query.lower().split())

        for result in results["metadatas"][0]:
            caption = result.get("ai_caption", "").lower()
            tags = result.get("ai_tags", "").lower()

            keyword_match = len(
                query_words & (set(caption.split()) | set(tags.split(",")))
            ) / len(query_words)

            # 통합 스코어
            result["score"] = 0.7 * result["score"] + 0.3 * keyword_match

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
```

---

### U-019: GUI 레이어 뷰어

```jsx
// frontend/src/components/LayerViewer.jsx
export const LayerViewer = ({ psdFile }) => {
  const [layers, setLayers] = useState([]);
  const [selectedLayer, setSelectedLayer] = useState(null);

  useEffect(() => {
    // 레이어 목록 로드
    window.electron.readLayers(psdFile).then(setLayers);
  }, [psdFile]);

  return (
    <div className="layer-grid">
      {layers.map(layer => (
        <div key={layer.id} className="layer-card">
          <img src={layer.thumbnail} />
          <div className="layer-info">
            <h4>{layer.name}</h4>
            <p>{layer.ai_caption}</p>
            <div className="tags">
              {layer.ai_tags.split(',').map(tag => (
                <span className="tag">{tag}</span>
              ))}
            </div>
            <div className="meta">
              <span>Size: {layer.width}×{layer.height}</span>
              <span>Pos: ({layer.pos_x}, {layer.pos_y})</span>
              <span>Type: {layer.layer_type}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
```

---

## 🎯 성능 예측

### 처리량 예측 (1000 레이어 기준)

| 환경 | 모델 | 배치 크기 | 예상 시간 |
|------|------|----------|----------|
| **Windows RTX 3060 Ti** | Florence-2 | 10 layers | ~50분 |
| **MacBook M5 32GB** | Qwen2.5-VL | 32 layers | ~30분 |

**병목 구간:**
1. 레이어 렌더링: ~0.1초/layer (CPU)
2. Vision 분석: ~0.8초/layer (GPU/NPU)
3. 벡터 인덱싱: ~0.05초/layer (DB)

**최적화 전략:**
- 멀티프로세싱: 렌더링 4 workers
- 배치 처리: Vision 분석 병렬화
- 캐싱: 동일 레이어 재분석 방지

---

## 📚 최종 디렉토리 구조

```
ImageParser/
├── backend/
│   ├── vision/
│   │   ├── layer_renderer.py      # 레이어 PNG 렌더링
│   │   ├── layer_analyzer.py      # 레이어 단위 Vision 분석
│   │   ├── analyzer_florence.py   # Windows용 (Florence-2)
│   │   ├── analyzer_qwen.py       # Mac용 (Qwen2.5-VL)
│   │   └── prompt_templates.py
│   ├── vector/
│   │   ├── layer_indexer.py       # 레이어 단위 인덱싱
│   │   └── layer_searcher.py      # 하이브리드 검색
│   └── pipeline/
│       └── ingest_engine.py       # 파일 + 레이어 통합 파이프라인
├── chroma_db/
│   ├── file_library/              # 기존 파일 단위
│   └── layer_library/             # NEW 레이어 단위
└── docs/
    ├── unified_architecture.md    # 이 문서
    └── layer_schema_spec.md       # 레이어 스키마 상세
```

---

## 🚀 다음 단계

### 1. 의존성 설치
```powershell
# Windows
python backend/setup/installer.py --install
pip install transformers timm einops

# MacBook M5
brew install ollama
ollama pull qwen2.5-vl:7b
pip install ollama chromadb sentence-transformers
```

### 2. 레이어 렌더링 테스트
```python
python -c "
from backend.vision.layer_renderer import LayerRenderer
renderer = LayerRenderer()
layers = renderer.render_all_layers('test.psd')
print(f'Rendered {len(layers)} layers')
"
```

### 3. Phase 4 시작
```powershell
/unit-start  # U-015: 레이어 렌더링 및 기반 구축
```

---

## 💡 핵심 요약

**이 시스템의 본질:**
> "PSD 파일을 원자(레이어) 단위로 분해하고, 각 원자에 Qwen(서기)과 CLIP(목격자)의 기록을 모두 새긴 뒤, ChromaDB라는 도서관에 보관하여 어떤 질문에도 즉시 답할 수 있게 만드는 것"

**3가지 핵심 가치:**
1. **레이어 단위 분석**: 파일이 아닌 레이어별 세밀한 인덱싱
2. **하이브리드 검색**: 감각(CLIP) + 지성(Qwen) + 필터(메타데이터)
3. **환경 최적화**: Windows(Florence) / Mac(Qwen) 모두 지원

---

**작성일**: 2026-02-06
**통합 버전**: v1.0
**다음 문서**: `layer_implementation_guide.md`
