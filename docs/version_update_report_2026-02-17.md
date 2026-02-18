# ImageParser 버전 업데이트 작업 내역 (2026-02-17)

## 범위
- 대상 버전: `v3.6.0.20260217_13` ~ `v3.6.0.20260217_18`
- 대상 이슈: VLM 단계 정체, MPS/CPU 경로 안정성, 배치 동작 복구, Qwen-VL padding 경고

## 버전별 변경 요약

| 버전 | 커밋 | 유형 | 핵심 변경 |
|---|---|---|---|
| v3.6.0.20260217_13 | `74c78b7` | chore | 버전 표기 갱신 |
| v3.6.0.20260217_15 | `0593fe5` | chore | 버전 표기 갱신 |
| v3.6.0.20260217_16 | `bd4d8c2` | chore | 버전 표기 갱신 |
| v3.6.0.20260217_17 | `55cc6af` | chore | 버전 표기 갱신 |
| v3.6.0.20260217_18 | `22ed9bb` | chore | 버전 표기 갱신 |

## 기능/버그 수정 상세

### 1) `219ca64` - `fix: stabilize VLM phase on CPU and fail fast on timeout`
- 파일: `backend/pipeline/ingest_engine.py`, `backend/vision/analyzer.py`
- 목적:
  - VLM 단계에서 긴 무응답 구간 발생 시 즉시 원인 파악 가능하도록 fail-fast 강화
  - CPU 경로에서 진행률 정체를 줄이기 위한 안정화
- 핵심:
  - 타임아웃 처리/오류 경로 강화
  - CPU 경로에서의 안정적 진행 로직 보강

### 2) `7cda338` - `fix: fail fast on macOS CPU fallback for VLM`
- 파일: `backend/vision/analyzer.py`, `config.yaml`
- 목적:
  - macOS에서 VLM이 의도치 않게 CPU로 내려갔을 때 조용히 느려지는 문제 방지
- 핵심:
  - `fail_if_cpu_on_macos` 정책 적용
  - MPS 미사용 상태를 경고가 아닌 실패로 처리 가능하게 함

### 3) `01b61cf` - `fix: remove forced single-batch VLM path on MPS`
- 파일: `backend/pipeline/ingest_engine.py`, `backend/vision/analyzer.py`
- 목적:
  - MPS에서 강제로 `batch=1`로 고정되던 경로 제거
  - Adaptive batching이 실제로 증가하도록 복구
- 핵심:
  - MPS 강제 단일 배치 분기 제거
  - VLM 배치 경로를 tier/adaptive 설정에 맞게 재활성화

### 4) `7334a22` - `fix: enforce left padding for Qwen-VL generation`
- 파일: `backend/vision/analyzer.py`
- 목적:
  - `A decoder-only architecture ... right-padding was detected` 경고 제거
  - Qwen-VL 배치 생성 일관성 개선
- 핵심:
  - Qwen2/Qwen3 processor 로드 후 tokenizer `padding_side='left'` 강제
  - `pad_token` 미설정 시 `eos_token`으로 보정

## 이번 사이클 테스트 관찰 요약

### A. 배치 증가 복구 확인
- 실행 조건: `--discover ... --no-skip` (교장실 12개 파일)
- 관찰 로그:
  - `ADAPTIVE:vlm B:2→3`
  - `ADAPTIVE:vlm B:3→5`
- 결론: VLM adaptive 배치 증가 경로 정상 복구

### B. 정체 구간 상태
- 과거: `2/12` 부근 장시간 무응답 반복
- 현재: 배치 단위 전진 확인, 완전 정지 패턴은 재현되지 않음
- 비고: 파일/배치당 추론 시간 자체는 여전히 큼(모델 계산량 영향)

### C. padding 경고 상태
- 수정 후 테스트에서 `right-padding detected` 경고 재발 없음(관찰 구간 기준)
- 결론: Qwen-VL padding 설정 패치 반영됨

## 남은 과제
- 속도 이슈는 환경(MPS/메모리)보다 모델 계산량/2-stage 비용 비중이 큼
- 다음 후보:
  - 2-stage 유지 + context 길이 동적 제한
  - 1-stage 옵션 병행(A/B)
  - 연속 작업 시 VLM 상주 전략 검토

## 참고 파일
- `backend/pipeline/ingest_engine.py`
- `backend/vision/analyzer.py`
- `frontend/src/components/StatusBar.jsx`
- `config.yaml`
