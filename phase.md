# phase.md - Imagine 개발 로드맵

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
- [x] Triaxis 검색 (VV+MV+FTS, RRF 결합)
- [x] Tier 시스템 (standard/pro/ultra)
- [x] 크로스 플랫폼 VLM 폴백 체인 (mlx→transformers→ollama)

### Phase 4: Electron GUI + 클라이언트-서버 아키텍처
- [x] React 19 + Electron 40 프론트엔드
- [x] 파일 브라우저 + 가상 스크롤 그리드
- [x] Triaxis 검색 UI + 필터
- [x] 메타데이터 모달 (AI 분석 + 사용자 태그/노트/평점)
- [x] 이미지 검색 (단일/다중, AND/OR 모드)
- [x] i18n (한국어/영어)
- [x] 등록 폴더 자동 스캔 + Resume Dialog
- [x] 설정 모달
- [x] Electron/Web 듀얼 모드 (IPC ↔ HTTP 브리지)
- [x] FastAPI 서버 (JWT 인증, SPA 서빙)
- [x] 역할 기반 접근 제어 (admin/user)

### Phase 4.5: 분산 워커 시스템 (v10.x)
- [x] 워커 데몬 (Prefetch 풀 + 하트비트 + 명령 피기백)
- [x] 워커 세션 관리 API (connect/heartbeat/disconnect/admin)
- [x] 워커 토큰 원클릭 셋업 스크립트
- [x] Phase별 배치 처리 (process_batch_phased — 워커 내부)
- [x] Phase 간 모델 언로드 (VLM→SigLIP2→Qwen3-Embedding 순차)
- [x] VV/MV 서브배치 추론 (encode_image_batch, encode_batch)
- [x] Phase별 처리 속도 UI (files/min per phase)
- [x] 멀티 워커 계정 생성 (Admin Panel)
- [x] Admin 워커 모니터링 (aggregate/per-worker 처리량)
- [x] 작업 큐 관리 (Job 생성/claim/완료/실패)

### Phase 4.6: 서버 외부 접속 (v10.5)
- [x] LAN IP 자동 감지 (os.networkInterfaces)
- [x] ServerInfoPanel (서버 정보 드롭다운 — Local/LAN/Tunnel URL)
- [x] QR 코드 생성 (qrcode.react — LAN/Tunnel URL)
- [x] CORS 완화 (cors_allow_all 옵션, JWT 보호)
- [x] Cloudflare Quick Tunnel (원클릭 인터넷 접속, 자동 다운로드)

### 코드베이스 정리 (v10.4)
- [x] 레거시 코드 삭제 (PostgreSQL, ChromaDB, Ollama parallel — 17파일, 4575줄)
- [x] 설계 근거 문서화 (9개 아키텍처 결정)
- [x] 트러블슈팅 문서화 (MC 데이터 손실, MV import, IPC 세션, GPU 경합)

---

## 진행 예정 Phase

### Phase 5: UI/UX 개선 (우선순위 1)
- [x] 로컬 폴더 동기화 (DB ↔ 디스크 정합성 — 이동/삭제/신규 감지)
- [x] 이미지 원본 다운로드 (서버 모드 — Web 브라우저에서 원본 파일 다운로드)
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

### Phase 8: 패키징 및 배포 (우선순위 4)
- [ ] Python 임베디드 런타임 번들링 (pyinstaller/embedded Python)
- [ ] Windows NSIS 인스톨러
- [ ] macOS DMG 빌드 (코드 서명)
- [ ] electron-updater 자동 업데이트
- [ ] 첫 실행 가이드 (모델 다운로드 위자드)
- [ ] 포터블 모드 (USB 실행)

### Phase 9: 데이터 관리 + 협업 (우선순위 5)
- [ ] DB 자동 백업 (스케줄 + 보존 정책)
- [ ] DB 수동 백업/복구 UI
- [ ] 파일 매칭/재링크 UI 개선 (매칭 시각화, 선택적 적용)
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
| 4 | Electron GUI + 서버 | 완료 |
| 4.5 | 분산 워커 시스템 | 완료 |
| 4.6 | 서버 외부 접속 | 완료 |
| 5 | UI/UX 개선 | 예정 |
| 6 | 검색 고도화 | 예정 |
| 7 | 성능 최적화 | 예정 |
| 8 | 패키징/배포 | 예정 |
| 9 | 데이터 관리 + 협업 | 예정 |
