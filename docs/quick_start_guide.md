# Quick Start Guide - ImageParser v3.1.1

빠른 시작 가이드 - 플랫폼별 최적 설정

---

## 🚀 5분 만에 시작하기

### 1. 플랫폼 확인

```bash
# 현재 플랫폼과 권장 설정 확인
python -m backend.utils.platform_detector
```

**예상 출력 (Windows):**
```
OS: Windows
Optimal Backend: ollama
Optimal Batch Size: 1
```

---

### 2. 단일 파일 처리

```bash
# AUTO 모드로 단일 이미지 처리
python backend/pipeline/ingest_engine.py \
  --file "path/to/image.psd"
```

**소요 시간 (Ultra tier):**
- Windows: ~30-70초
- Mac/Linux (vLLM): ~6-10초

---

### 3. 여러 파일 처리

```bash
# AUTO 모드로 여러 파일 배치 처리
python backend/pipeline/ingest_engine.py \
  --files '["file1.psd", "file2.png", "file3.jpg"]' \
  --batch-size auto
```

**권장 batch_size:**
- **Windows**: 1 (순차 처리)
- **Mac/Linux**: 16 (병렬 처리)

---

### 4. 폴더 전체 처리

```bash
# 폴더 내 모든 이미지 자동 탐색 및 처리
python backend/pipeline/ingest_engine.py \
  --discover "C:\path\to\assets" \
  --batch-size auto
```

---

## ⚙️ 플랫폼별 설정

### Windows 사용자

**권장 설정:**
```yaml
# config.yaml
ai_mode:
  override: ultra  # 또는 standard/pro
  tiers:
    ultra:
      vlm:
        backend: auto  # Ollama 자동 선택
```

**특징:**
- ✅ 안정적이고 신뢰성 높음
- ⚠️ 순차 처리만 지원 (batch_size=1)
- ⚠️ Mac/Linux 대비 8.5배 느림

**명령어:**
```bash
# 순차 처리 (권장)
python backend/pipeline/ingest_engine.py \
  --discover "assets" \
  --batch-size 1
```

---

### Mac 사용자

**최고 성능 설정 (vLLM):**
```bash
# vLLM 설치
pip install vllm

# config.yaml은 AUTO로 유지
# vLLM이 설치되면 자동으로 선택됨
```

**특징:**
- ✅ 8.5배 빠른 배치 처리
- ✅ 16개 이미지 동시 처리 가능
- ✅ PagedAttention 메모리 최적화

**명령어:**
```bash
# 배치 처리 (권장)
python backend/pipeline/ingest_engine.py \
  --discover "assets" \
  --batch-size 16
```

**대안 (Ollama):**
```bash
# vLLM 없이 Ollama 사용
brew install ollama
ollama serve
ollama pull qwen3-vl:8b

# config.yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: ollama  # 명시적 지정
```

---

### Linux 사용자

**최고 성능 설정 (vLLM):**
```bash
# vLLM 설치 (CUDA 지원)
pip install vllm

# GPU 확인
nvidia-smi
```

**명령어:**
```bash
# 배치 처리 (권장)
python backend/pipeline/ingest_engine.py \
  --discover "assets" \
  --batch-size 16
```

---

## 📊 성능 비교

### Ultra Tier 기준 (10개 이미지)

| 플랫폼 | 백엔드 | Batch Size | 총 시간 | 파일당 |
|--------|--------|-----------|---------|--------|
| Windows | Ollama | 1 | ~510초 | ~51초 |
| Mac | vLLM | 16 | ~60초 | ~6초 |
| Mac | Ollama | 1 | ~510초 | ~51초 |
| Linux | vLLM | 16 | ~60초 | ~6초 |

**결론**: Mac/Linux + vLLM = **8.5배 빠름**

---

## 🎯 Tier 선택 가이드

### Standard Tier
**대상**: 빠른 프로토타이핑, 개발 환경
```yaml
ai_mode:
  override: standard
```

**특징:**
- 모델: moondream2 (경량)
- VRAM: ~4GB
- 속도: 12초 (10개)
- 품질: 기본

**사용 시기:**
- 빠른 반복 개발
- 테스트 및 디버깅
- VRAM 부족한 환경

---

### Pro Tier
**대상**: 균형잡힌 품질과 속도
```yaml
ai_mode:
  override: pro
```

**특징:**
- 모델: Qwen3-VL-4B
- VRAM: ~12GB
- 속도: 38초 (10개)
- 품질: 우수

**사용 시기:**
- 일반적인 프로덕션 환경
- 중간 품질 필요
- VRAM 12-16GB

---

### Ultra Tier
**대상**: 최고 품질, 프로덕션
```yaml
ai_mode:
  override: ultra
```

**특징:**
- 모델: Qwen3-VL-8B
- VRAM: ~20GB
- 속도:
  - Windows: 510초 (10개)
  - Mac/Linux: 60초 (10개, vLLM)
- 품질: 최고

**사용 시기:**
- 최종 프로덕션 데이터
- 최고 품질 필요
- VRAM 20GB+

---

## 💡 실전 예시

### 예시 1: 개발 중 빠른 테스트 (Windows)

```bash
# Standard tier로 빠르게 테스트
# config.yaml에서 override: standard 설정

python backend/pipeline/ingest_engine.py \
  --file "test.psd"
```

**소요 시간**: ~12초

---

### 예시 2: 프로덕션 데이터 처리 (Mac + vLLM)

```bash
# Ultra tier로 최고 품질
# config.yaml에서 override: ultra 설정

python backend/pipeline/ingest_engine.py \
  --discover "production_assets" \
  --batch-size 16
```

**100개 파일 예상 시간**: ~10분

---

### 예시 3: 대량 처리 (Windows)

```bash
# Ultra tier, 순차 처리
python backend/pipeline/ingest_engine.py \
  --discover "large_dataset" \
  --batch-size 1
```

**100개 파일 예상 시간**: ~85분
**권장**: Mac/Linux 서버로 이전 고려

---

## 🔧 문제 해결

### 문제: "Ollama server is not running"

**해결:**
```bash
# Ollama 시작
ollama serve

# 다른 터미널에서 확인
ollama list
```

---

### 문제: "vLLM is not supported on Windows"

**해결:**
- vLLM은 Windows를 지원하지 않습니다
- **옵션 1**: Ollama 사용 (순차 처리)
- **옵션 2**: WSL2 + vLLM (고급 사용자)
- **옵션 3**: Mac/Linux 서버로 이전

---

### 문제: 배치 처리가 느림 (Windows)

**확인:**
```bash
# 플랫폼 권장사항 확인
python -m backend.utils.platform_detector
```

**해결:**
- Windows + Ollama는 batch_size=1이 최적
- 병렬 처리 시 오히려 느려짐 (0.6x)
- `--batch-size 1` 사용 권장

---

### 문제: VRAM 부족

**해결:**
- **Tier 낮추기**: ultra → pro → standard
- **Batch size 줄이기**: 16 → 8 → 4 → 1
- **모델 언로드**: 처리 후 모델 메모리 해제

---

## 📚 추가 자료

- [플랫폼별 최적화 가이드](platform_optimization.md) - 상세 설정
- [V3.1 릴리즈 노트](V3.1.md) - 3-Tier 시스템
- [Ollama 배치 분석](ollama_batch_processing_analysis.md) - 성능 벤치마크

---

## 🎓 Best Practices

### 1. 항상 AUTO 모드 사용

```yaml
# config.yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: auto  # 플랫폼 자동 감지
```

### 2. 적절한 Tier 선택

- **개발**: Standard
- **테스트**: Pro
- **프로덕션**: Ultra

### 3. 플랫폼별 최적화

- **Windows**: batch_size=1
- **Mac/Linux**: batch_size=16 (vLLM)

### 4. 대량 처리 시

- 소규모 테스트 먼저 (10개)
- 에러 확인
- 전체 처리

### 5. 성능 모니터링

```bash
# 처리 시간 기록
python backend/pipeline/ingest_engine.py \
  --discover "assets" \
  --batch-size auto \
  2>&1 | tee process.log
```

---

**버전**: v3.1.1
**최종 업데이트**: 2026-02-09
