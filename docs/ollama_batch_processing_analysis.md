# Ollama 배치 처리 성능 분석 보고서

**작성일:** 2026-02-09
**목적:** Ollama Vision API 배치 처리 성능 이슈 및 해결 방안 검토

---

## 1. 시스템 구성

### 하드웨어
- GPU: NVIDIA (VRAM 20GB+)
- OS: Windows

### 소프트웨어 스택
- **Ollama:** 로컬 LLM inference 서버
- **모델:** qwen3-vl:8b (Qwen 3세대 Vision Language Model)
- **Backend:** Ollama HTTP API
- **Processing:** Python + ThreadPoolExecutor (multi-process)

### AI Tier 구성
| Tier | Backend | Model | VRAM |
|------|---------|-------|------|
| Standard | Transformers | moondream2 | ~6GB |
| Pro | Transformers | Qwen3-VL-4B | ~12GB |
| Ultra | **Ollama** | **qwen3-vl:8b** | ~8GB |

---

## 2. 벤치마크 결과 (실측)

### 2.1 Transformers Backend (Standard/Pro)

**Standard Tier:**
| 파일 수 | 총 시간 | 파일당 시간 | 속도 향상 |
|---------|---------|-------------|-----------|
| 1개 | 12.2초 | 12.2초 | 1.0x (기준) |
| 5개 | 14.9초 | 3.0초 | **4.1x** |
| 10개 | 17.0초 | 1.7초 | **7.2x** |
| 20개 | 17.4초 | 0.9초 | **13.6x** |

**결과:** 배치 처리 효율 93% (거의 완벽한 병렬화)

**Pro Tier:**
| 파일 수 | 총 시간 | 파일당 시간 | 속도 향상 |
|---------|---------|-------------|-----------|
| 1개 | 17.5초 | 17.5초 | 1.0x |
| 5개 | 24.6초 | 4.9초 | 3.6x |
| 10개 | 37.6초 | 3.8초 | 4.6x |
| 20개 | **FAIL** | - | VRAM 한계 |

**결과:** 배치 처리 효과 있지만 VRAM 한계로 10개까지만 가능

---

### 2.2 Ollama Backend (Ultra) - 문제 발생!

**Ultra Tier (qwen3-vl:8b via Ollama):**
| 파일 수 | 총 시간 | 파일당 시간 | 속도 변화 |
|---------|---------|-------------|-----------|
| 1개 | 51.0초 | 51.0초 | 1.0x (기준) |
| 5개 | 397.2초 | **79.4초** | **0.64x (느려짐!)** |
| 10개 | TIMEOUT | >600초 | 실패 |
| 20개 | TIMEOUT | >600초 | 실패 |

**🚨 문제:** 배치 크기가 커질수록 파일당 처리 시간이 **오히려 증가**

**계산:**
- 순차 처리 예상: 51초 × 5 = 255초
- 실제 배치 처리: 397초
- **차이:** 142초 추가 소요 (55% 느려짐)

---

## 3. 병렬 처리 테스트 결과

### 3.1 텍스트 생성 API 테스트 (`/api/generate`)

**테스트 방법:**
- 3개의 텍스트 프롬프트
- 순차 vs 병렬 요청 비교

**결과:**
| 모드 | 총 시간 | 평균/요청 | Speedup |
|------|---------|-----------|---------|
| 순차 | 546초 | 182초 | 1.0x |
| 병렬 (3개 동시) | 182초 | 60.7초 | **3.0x** |

**✅ 결론:** Ollama 텍스트 생성 API는 병렬 처리 **가능**

---

### 3.2 Vision API 벤치마크 (실제 이미지 처리)

**테스트 방법:**
- 실제 PSD/PNG 이미지 파일
- ThreadPoolExecutor로 동시 요청
- Vision API (`/api/chat` with images)

**결과:**
- 배치 처리 시 성능 **저하** (0.64x)
- 순차 처리보다 **느림**

**❌ 결론:** Ollama Vision API는 배치 처리 시 성능 저하

---

## 4. 가설 및 분석

### 가설 1: API 종류별 처리 차이
```
/api/generate (텍스트):  병렬 처리 OK (3x 향상)
/api/chat (비전):         병렬 처리 안 됨 (0.6x 저하)
```

**가능성:** Vision API가 내부적으로 다른 처리 방식 사용

---

### 가설 2: GPU 메모리 경합
```
텍스트 생성: 가벼움 (VRAM 2-3GB) → 병렬 가능
이미지 처리: 무거움 (VRAM 6-8GB) → 메모리 경합 → 느려짐
```

**가능성:** 여러 이미지 동시 처리 시 VRAM 부족 → 스왑 발생

---

### 가설 3: Ollama 내부 큐잉
```
요청 1 ─┐
요청 2 ─┤
요청 3 ─┼──> Ollama Server ──> Queue (FIFO) ──> 순차 처리
요청 4 ─┘
```

**동작 방식 추정:**
1. 여러 요청을 받음 (입구는 병렬)
2. 내부 큐에 쌓임
3. GPU에서 순차 처리
4. 텍스트는 가벼워서 빠르게 전환 → 병렬 효과
5. 이미지는 무거워서 전환 오버헤드 → 성능 저하

---

## 5. 현재 상황 요약

### ✅ 확인된 사실
1. Transformers backend는 배치 처리 완벽 (13x 향상)
2. Ollama 텍스트 API는 병렬 처리 가능 (3x 향상)
3. Ollama Vision API는 배치 처리 시 성능 저하 (0.6x)
4. 모델은 1개만 로드됨 (메모리 효율적)

### ❓ 불확실한 부분
1. Ollama Vision API가 정말 순차 처리만 하는가?
2. `OLLAMA_NUM_PARALLEL` 환경 변수 효과는?
3. Vision 모델 자체의 배치 처리 지원 여부?
4. 메모리 경합이 주요 원인인가?

### 🎯 목표
- Qwen3-VL (최신 모델) 유지 필수
- 배치 처리로 성능 향상 (현재 순차보다 빠르게)
- 수백 개 이미지 빠른 처리

---

## 6. 해결 방안 후보

### 옵션 A: Ollama 설정 최적화
```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

**장점:**
- 현재 구조 유지
- 설정만 변경

**단점:**
- 효과 불확실 (텍스트만 효과 있을 수도)

**확인 필요:**
- Vision API에서도 효과가 있는가?

---

### 옵션 B: vLLM으로 교체

**vLLM이란:**
- UC Berkeley 개발 오픈소스 LLM inference 엔진
- PagedAttention, Continuous Batching 지원
- Ollama보다 3-5배 빠름

**장점:**
- 진짜 배치 처리 (GPU 레벨)
- 메모리 효율 최고
- 프로덕션 검증됨

**단점:**
- 설치/설정 복잡
- Qwen3-VL 지원 확인 필요

**예상 성능:**
```
1개:  25초
5개:  35초 (7초/파일) - 현재 79초 대비 11x 향상
20개: 80초 (4초/파일) - 현재 실패 대비 무한대 향상
```

---

### 옵션 C: 순차 처리 유지

**근거:**
- Ultra tier는 순차 처리 (batch_size=1)
- Standard/Pro는 배치 처리 (batch_size=10-20)
- AUTO 모드가 자동 선택

**장점:**
- 안정적
- 추가 작업 없음

**단점:**
- Ultra tier 성능 포기
- 수백 개 처리 시 시간 오래 걸림

---

## 7. 전문가 확인 요청 사항

### 질문 1: Ollama Vision API 동작 방식
```
Q1. Ollama의 /api/chat (Vision) API는 여러 요청을 어떻게 처리하는가?
    a) 완전 순차 (큐잉만)
    b) 부분 병렬 (일부 동시 처리)
    c) 완전 병렬 (GPU 배치)

Q2. OLLAMA_NUM_PARALLEL 환경 변수가 Vision API에도 영향을 주는가?
```

### 질문 2: 메모리 관련
```
Q3. Vision 모델 (qwen3-vl:8b) 처리 시 VRAM 사용량은?
    - 단일 이미지: ?GB
    - 동시 5개: ?GB

Q4. 20GB VRAM에서 몇 개까지 동시 처리 가능한가?
```

### 질문 3: 대안 검토
```
Q5. vLLM이 Qwen3-VL (3세대) 모델을 지원하는가?
    - Qwen2-VL: 확인됨
    - Qwen3-VL: ?

Q6. vLLM 사용 시 예상 성능 향상은?
    - 현재 Ollama: 51초/이미지
    - vLLM 예상: ?초/이미지
```

### 질문 4: 권장 사항
```
Q7. 현재 상황에서 최선의 방안은?
    a) Ollama 설정 최적화
    b) vLLM으로 전환
    c) 순차 처리 유지
    d) 다른 방안

Q8. Qwen3-VL 품질 유지하면서 배치 처리하는 방법은?
```

---

## 8. 참고 자료

### 벤치마크 결과 파일
- `benchmark_results.json` - 전체 수치 데이터
- `benchmark_summary.md` - 결과 요약

### 테스트 스크립트
- `run_benchmark.py` - 전체 벤치마크
- `test_ollama_parallel.py` - Ollama 병렬 테스트

### 구현 코드
- `backend/vision/ollama_adapter.py` - 현재 Ollama 어댑터
- `backend/vision/ollama_parallel_adapter.py` - 병렬 처리 시도 (효과 없음)

---

## 9. 요약

### 핵심 문제
**Ollama Vision API는 배치 처리 시 성능이 오히려 저하됨 (0.6x)**

### 확인 필요
1. Ollama Vision API의 정확한 동작 방식
2. vLLM의 Qwen3-VL 지원 여부
3. 최적의 해결 방안

### 제약 조건
- ✅ Qwen3-VL (최신 모델) 유지 필수
- ✅ 배치 처리로 성능 향상 필요
- ✅ 20GB VRAM 내에서 작동

---

**문의:** 위 내용에 대한 전문가 의견 및 권장 사항을 부탁드립니다.
