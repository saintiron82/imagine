# Phase 4 시작 가이드 (Getting Started)

## 📋 작업 시작 전 체크리스트

### ✅ 1단계: 환경 확인

```powershell
# Python 환경 확인
python --version
# 예상: Python 3.11.9

# GPU 확인
nvidia-smi
# 예상: RTX 3060 Ti, CUDA 12.6

# 가상 환경 활성화 확인
python -c "import sys; print(sys.executable)"
# 예상: C:\Users\saint\ImageParser\.venv\Scripts\python.exe
```

**체크:**
- [x] Python 3.11.9
- [x] NVIDIA RTX 3060 Ti (8GB)
- [x] 가상 환경 활성화됨

---

### ⚠️ 2단계: 핵심 의존성 설치 (CRITICAL)

**현재 상태:**
```json
{
  "torch": false,              // ❌ 미설치
  "chromadb": false,           // ❌ 미설치
  "sentence-transformers": false, // ❌ 미설치
  "pillow": false              // ❌ 미설치
}
```

**설치 명령어:**

```powershell
# 방법 1: 자동 설치 스크립트 (권장)
python backend/setup/installer.py --install
python backend/setup/installer.py --download-model

# 방법 2: 수동 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install chromadb sentence-transformers pillow transformers timm einops

# 설치 검증
python backend/setup/installer.py --check
```

**예상 설치 시간:** 5-10분 (인터넷 속도에 따라)
**디스크 공간 요구:** ~5GB (PyTorch + CLIP 모델)

---

### 📦 3단계: Phase 4 전용 의존성 추가

**requirements.txt 업데이트:**

```bash
# Phase 4 추가 라이브러리
transformers>=4.40.0    # HuggingFace Transformers (Florence-2)
timm>=0.9.0            # PyTorch Image Models
einops>=0.7.0          # Tensor 연산 헬퍼
```

**설치:**
```powershell
pip install transformers>=4.40.0 timm>=0.9.0 einops>=0.7.0
```

**검증:**
```powershell
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; print('Transformers OK')"
```

---

### 🧪 4단계: 기존 시스템 검증

**Phase 1-3이 정상 작동하는지 확인:**

```powershell
# 1. 스키마 import 테스트
python -c "from backend.parser.schema import AssetMeta; print('Schema OK')"

# 2. 파서 테스트
python test_image_parser.py

# 3. 기본 파이프라인 테스트 (의존성 설치 후)
python backend/pipeline/ingest_engine.py --file "test_assets/sample.png"

# 4. 벡터 검색 테스트
python backend/cli_search.py "test query"
```

**예상 결과:**
- ✅ 모든 테스트 PASS
- ✅ `output/json/sample.json` 생성됨
- ✅ `chroma_db/` 업데이트됨

**문제 발생 시:**
- `docs/troubleshooting.md` 참조
- `/troubleshoot` 명령어로 문제 기록

---

## 🚀 Phase 4 작업 시작

### U-015 시작 준비

**1. 디렉토리 생성:**
```powershell
mkdir backend\vision
New-Item backend\vision\__init__.py -ItemType File
New-Item backend\vision\analyzer.py -ItemType File
```

**2. 유닛 개발 프로토콜 시작:**
```powershell
/unit-start
```

**3. 목표 선언:**
```markdown
## U-015: Vision 모듈 기반 구축
### 1. 목표
- Florence-2 모델을 로컬에서 실행할 수 있는 VisionAnalyzer 클래스 구현
- 단일 이미지에서 캡션 생성 검증

### 완료 조건
- [ ] Florence-2 모델 로드 성공 (<10초)
- [ ] 테스트 이미지에서 캡션 생성 (<1초)
- [ ] 캡션 길이 10자 이상, 의미 있는 내용
```

---

## 📚 참고 자료

### Florence-2 문서
- **HuggingFace**: https://huggingface.co/microsoft/Florence-2-large
- **논문**: https://arxiv.org/abs/2311.06242
- **예제 코드**: https://huggingface.co/microsoft/Florence-2-large#usage

### 예제 코드 스니펫

```python
# Florence-2 기본 사용법
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

# 이미지 분석
image = Image.open("sample.png")
prompt = "<DETAILED_CAPTION>"

inputs = processor(text=prompt, images=image, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

print(result)
# 출력 예: "A fantasy knight character with glowing sword..."
```

---

## 🎯 작업 타임라인

### Week 1: Foundation & Integration
```
Day 1 (오늘):
  [x] Phase 4 명세서 작성 (완료)
  [x] 체크리스트 업데이트 (완료)
  [ ] 의존성 설치
  [ ] U-015 시작

Day 2-3:
  [ ] U-015 완료 (Vision 모듈)
  [ ] U-016 완료 (프롬프트 엔지니어링)

Day 4-5:
  [ ] U-017 완료 (파이프라인 통합)
  [ ] E2E 테스트

Weekend:
  [ ] 문서화 및 troubleshooting 기록
```

### Week 2: Enhancement & UI
```
Day 1-2:
  [ ] U-018 (검색 확장)

Day 3-4:
  [ ] U-019 (GUI 통합)

Day 5:
  [ ] Phase 4 완료 검증
  [ ] 회고 작성
```

---

## ⚡ 빠른 시작 명령어 요약

```powershell
# 1. 의존성 설치
python backend/setup/installer.py --install
pip install transformers timm einops

# 2. 디렉토리 생성
mkdir backend\vision

# 3. 작업 시작
/unit-start

# 4. 첫 코드 작성
# backend/vision/analyzer.py 편집 시작
```

---

## 🆘 문제 해결

### 설치 오류 발생 시
```powershell
# PyTorch 재설치 (CUDA 12.1 버전)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers 버전 확인
pip show transformers
# 버전이 4.40.0 미만이면 업그레이드
pip install --upgrade transformers
```

### GPU 인식 안 될 때
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

### 모델 다운로드 느릴 때
- HuggingFace 미러 사용 고려
- 모델 수동 다운로드 후 로컬 경로 지정

---

**준비 완료!** 이제 `/unit-start`로 U-015를 시작하세요.
