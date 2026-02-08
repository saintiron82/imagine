# 3-Axis Multimodal Pipeline Roadmap

본 프로젝트는 이미지 데이터를 단순한 파일이 아닌, **3가지 차원(Axis)의 정보**로 분해하여 지능형 에이전트가 활용할 수 있는 데이터베이스를 구축하는 것을 목표로 합니다.

## 🏗️ Core Architecture: 3-Axis Data Decomposition

### 1. Structural Axis (구조적 데이터) - ✅ Phase 1 완료
*   **Source**: PSD Layers, Metadata
*   **Tech**: `psd-tools` (Python)
*   **Data**: 
    *   계층 구조(Layer Tree), 좌표(Coordinates), 투명도(Opacity)
    *   텍스트 내용(Text Content), 폰트 정보
    *   파일명, 해상도, 포맷
*   **Role**: 에이전트가 이미지를 "편집"하거나 "구성 요소"를 파악하는 기초 자료.

### 2. Latent Axis (잠재적/시각적 데이터) - ✅ Phase 2 완료 (파일 단위)
*   **Source**: Composite Image (Thumbnail)
*   **Tech**: `CLIP-ViT-L-14` (OpenAI/HuggingFace)
*   **Data**: 
    *   768차원 고밀도 벡터 (Embedding)
*   **Role**: "느낌", "분위기", "유사성" 기반의 모호한 검색 지원. (예: "불타는 검" -> 시각적 유사도 매칭)

### 3. Descriptive Axis (서술적/의미적 데이터) - 🚧 Phase 3 진행 예정
*   **Source**: Composite Image
*   **Tech**: `Qwen-VL` or `Florence-2` (Vision Language Model)
*   **Data**: 
    *   AI가 생성한 상세 캡션 (Caption)
    *   객체 태그 (Tagged Objects: "Knight", "Cape", "Sword")
    *   색감 및 조명 묘사 ("Dark Fantasy style", "Cinematic lighting")
*   **Role**: 키워드 기반 정밀 검색 및 에이전트의 "상황 인지" 능력 부여.

---

## 📅 Development Phases

### [Phase 1] Structural Parsing & Pipeline (완료)
- [x] **PSD Parser**: 레이어 및 텍스트 추출 엔진
- [x] **Meta Cleaner**: 데이터 정제(Garbage Filtering)
- [x] **Ingest Engine**: 대량 파일 처리 파이프라인
- [x] **Electron GUI**: 데이터 탐색기 및 결과 뷰어

### [Phase 2] Latent Vectorization (완료)
- [x] **Vector Indexer**: CLIP 모델 연동 및 임베딩 생성
- [x] **ChromaDB**: 로컬 벡터 저장소 구축
- [x] **Search System**: 텍스트-이미지 유사도 검색 구현

### [Phase 3] Basic Environment Setup (완료)
- [x] **Installer Script**: 의존성 자동 설치 (`torch`, `transformers`)
- [x] **Env Check**: Python/CUDA 환경 진단 모듈
- [x] **Settings UI**: 사용자 친화적 설치 메뉴 제공

### [Phase 4] Descriptive Vision Analysis (진행 중)
- [ ] **Vision Module**: Qwen/Florence-2 로컬 구동 (Axis 3 확보)
- [ ] **Caption Generator**: 이미지 상세 설명 생성
- [ ] **Data Fusion**: 구조(1) + 벡터(2) + 설명(3) 데이터를 하나의 DB 레코드로 통합

### [Phase 5] Optimization & Distribution (최종)
- [ ] **Layer-Level Indexing**: 레이어 단위 심층 분석
- [ ] **Full Packaging**: Vision Model을 포함한 배포 전략 수립
- [ ] **Installer**: 최종 사용자용 통합 설치 파일 (.exe) 제작
