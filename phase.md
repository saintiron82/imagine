# phase.md - ImageParser 개발 로드맵

## 완료된 Phase

### Phase 1: 구조적 파싱 (Structural Parsing)
- [x] PSD/PNG/JPG 파서 구현 (BaseParser, PSDParser, ImageParser)
- [x] 데이터 스키마 정의 (AssetMeta, LayerInfo, ParseResult)
- [x] 레이어 이름 정제 (Cleaner)
- [x] 썸네일 생성기
- [x] Ingest Pipeline 4단계 구축

### Phase 2: 시각 벡터화 (Visual Vectorization)
- [x] SigLIP2 VV (Visual Vector)
- [x] SQLite + sqlite-vec 마이그레이션 (ChromaDB/PostgreSQL 제거)
- [x] FTS5 전문 검색 인덱스 (FTS)

### Phase 3: 서술적 비전 (Descriptive Vision)
- [x] Qwen3-VL 2-Stage 캡션/태그/분류 생성
- [x] Qwen3-Embedding MV (Meaning Vector)
- [x] Triaxis 검색 (V+S+M, RRF 결합)
- [x] Tier 시스템 (standard/pro/ultra)
- [x] 플랫폼별 최적화 (AUTO 모드)

### Phase 4: Electron GUI
- [x] React 19 + Electron 40 프론트엔드
- [x] 파일 브라우저 + 가상 스크롤 그리드
- [x] Triaxis 검색 UI + 필터
- [x] 메타데이터 모달 (AI 분석 + 사용자 태그/노트/평점)
- [x] 이미지 검색 (단일/다중, AND/OR 모드)
- [x] i18n (한국어/영어)
- [x] 등록 폴더 자동 스캔
- [x] 설정 모달

---

## 진행 예정 Phase

### Phase 5: UI/UX 개선 (우선순위 1)
- [ ] 라이트박스 이미지 뷰어 (줌/패닝/좌우 네비게이션)
- [ ] 검색 히스토리 및 자동완성
- [ ] 결과 정렬 옵션 (관련도/날짜/파일명/평점)
- [ ] 뷰 모드 전환 (그리드/리스트/컴팩트)
- [ ] 드래그 앤 드롭 파일 처리
- [ ] 키보드 단축키 (Ctrl+K 검색 등)
- [ ] 사이드바 접기/펴기

### Phase 6: 검색 고도화 (우선순위 2)
- [ ] 고급 필터 (날짜 범위, 해상도 범위, 레이어 수)
- [ ] 필터 프리셋 저장/불러오기
- [ ] 북마크/컬렉션 시스템 (CRUD + JSON 내보내기)
- [ ] 스마트 컬렉션 (동적 필터 기반)
- [ ] "유사 이미지" 원클릭 검색
- [ ] 태그 일괄 편집
- [ ] 태그 클라우드 시각화

### Phase 7: 성능 최적화 (우선순위 3)
- [ ] 증분 인덱싱 고도화 (해시 기반 변경 감지)
- [ ] 멀티프로세스 병렬 파싱
- [ ] VLM 결과 캐싱
- [ ] 검색 응답 클라이언트 캐시
- [ ] 썸네일 WebP 전환
- [ ] 모델 자동 언로드 (idle timeout)
- [ ] SQLite WAL 모드 동시 읽기

### Phase 8: 패키징 및 배포 (우선순위 4)
- [ ] Windows NSIS 인스톨러 (Python 런타임 번들링)
- [ ] macOS DMG 빌드
- [ ] electron-updater 자동 업데이트
- [ ] 첫 실행 가이드 (모델 다운로드 위자드)
- [ ] 포터블 모드 (USB 실행)

### Phase 9: 협업 기능 (우선순위 5)
- [ ] DB 내보내기/가져오기 (메타데이터+벡터 패키징)
- [ ] 컬렉션/폴더 단위 부분 내보내기
- [ ] 이미지별 코멘트 히스토리
- [ ] 읽기 전용 공유 모드

---

## 마일스톤 요약

| Phase | 목표 | 상태 |
|-------|------|------|
| 1 | 구조적 파싱 | 완료 |
| 2 | 시각 벡터화 | 완료 |
| 3 | 서술적 비전 + Triaxis | 완료 |
| 4 | Electron GUI | 완료 |
| 5 | UI/UX 개선 | 예정 |
| 6 | 검색 고도화 | 예정 |
| 7 | 성능 최적화 | 예정 |
| 8 | 패키징/배포 | 예정 |
| 9 | 협업 기능 | 예정 |
