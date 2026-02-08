# 멀티모달 이미지 데이터화 시스템 명세서 (Multimodal Image Digitization Spec)

## 1. 프로젝트 개요
PSD, PNG, JPG 등 이미지 자산을 분석하여 AI가 이해할 수 있는 **멀티 레이어 데이터(Multi-Layer Data)**로 변환하고, 이를 벡터 DB에 구축하여 지능형 검색 및 맥락 파악이 가능한 시스템을 구축합니다.

## 2. 데이터 구성 명세 (Data Schema)
**하이브리드 저장 구조**를 채택합니다.
- **Raw Data (이미지)**: 로컬 파일 시스템에 저장 (`C:\Users\saint\ImageParser\Assets\...`)
- **Meta & Vector (DB)**: ChromaDB가 **벡터와 메타데이터(파일 경로, 작성자, 태그 등)를 통합 관리**합니다. 별도의 메타 DB(MySQL 등)가 필요 없습니다.

벡터 DB(ChromaDB)에 저장될 포인트(Point)의 규격입니다.

### [A] 벡터 영역 (Searchable Vectors)
AI가 검색의 '의미'를 파악하는 핵심 수치 데이터.
- **`vec_visual_composite`**: 전체 병합 이미지의 시각적 특징 (CLIP ViT 활용). 분위기, 색감, 구도 검색.
- **`vec_semantic_layers`**: 레이어 구조의 텍스트 특징 (BGE-Small 활용). 예: "Character/Body/Arm/Armor".
- **`vec_text_content`**: 이미지 내 텍스트 레이어의 실제 문자열 특징. 기획 의도 및 메모 검색.

### [B] 메타데이터 영역 (Payload)
에이전트 분석 및 필터링용 구조적 정보.
- **파일 정보**: `file_path` (로컬 절대 경로), `file_name`, `file_size`, `resolution`, `format`.
- **레이어 트리**: JSON 형태의 전체 계층 구조.
- **폰트 및 텍스트**: `used_fonts` 목록.
- **관리 정보**: `author`, `last_modified_at`, `project_tag`.

### [C] 참조 영역 (Reference)
- **`thumbnail_url`**: 로컬 섬네일 이미지 경로 (e.g., `file:///C:/...`).
- **`storage_type`**: `local_fs` (추후 S3 등으로 확장 가능).

## 3. 단계별 구축 계획 (Roadmap)

### [1단계] 데이터 추출 및 정규화 (Extraction & Normalization)
**목표**: PSD, PNG, JPG를 분석하여 공통된 'Standard Data Object'로 변환.
- **Universal Parser**:
    - **PSD**: 레이어 트리, 텍스트 레이어, 폰트 정보 추출, 렌더링(Composite).
    - **PNG/JPG**: 단일 레이어 구조로 취급하여 정규화. 메타데이터(Exif) 추출.
- **Preprocessing**: 레이어 이름 정규화(특수문자 제거), 텍스트 토큰화 준비.

### [2단계] 하이브리드 임베딩 (Hybrid Embedding)
**목표**: 추출된 데이터를 벡터 공간으로 투영.
- **Visual Embedding**: PSD에서 추출한 **Composite(병합된 미리보기)** 이미지를 사용합니다. (원본 PSD 직접 분석 X)
    - 최적화: $224 \times 224$ 크기로 리사이징하여 AI(CLIP)에 입력, 부하를 최소화합니다.
- **Semantic Embedding**: 레이어 경로 텍스트("Layer Path") → BGE 모델 → 텍스트 벡터.

### [3단계] 지식 데이터베이스 구축 (Vector DB Construction)
**목표**: ChromaDB 기반의 지식 저장소 구축.
- **Collection**: 'Asset_Library' 컬렉션 생성.
- **Schema**: Named Vectors (Visual, Semantic, Text 별도 필드) + Payload 인덱싱.
- **Search Logic**: HNSW 인덱싱 적용, 메타데이터 필터링(RBAC) 지원.

## 4. 상세 기술적 방안

### A. 파싱 및 정규화 기술
- **PSD**: `psd-tools` (심층 분석: 마스크, 조정 레이어 고려).
- **이미지(PNG/JPG)**: `Pillow`, `exifread`.

### B. 벡터화 모델 (Local Running)
- **Image**: OpenCLIP (ViT-B-32 or ViT-H-14).
- **Text**: BGE-M3 or BGE-Small-en/ko.

### C. DB 아키텍처 (ChromaDB)
- **Multi-Vector Support**: 하나의 포인트에 `image_vec`, `text_vec`을 동시에 저장.
- **Payload Indexing**: `project_tag`, `author` 등의 필드에 인덱스 생성하여 빠른 필터링 지원.