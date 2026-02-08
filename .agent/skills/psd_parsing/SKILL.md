---
description: PSD/이미지 파싱 기술 참조 가이드
---
# PSD 파싱 스킬 가이드

## 핵심 라이브러리: psd-tools

### 기본 사용법
```python
from psd_tools import PSDImage

psd = PSDImage.open('file.psd')

# 전체 이미지 렌더링
composite = psd.composite()
composite.save('output.png')

# 레이어 순회
for layer in psd.descendants():
    print(f"Name: {layer.name}, Type: {layer.kind}")
    print(f"Position: {layer.left}, {layer.top}")
    print(f"Size: {layer.width} x {layer.height}")
    print(f"Visible: {layer.visible}")
    print(f"Clipping: {layer.has_clip_layers()}")
```

### 레이어 종류 (layer.kind)
- `group`: 그룹/폴더
- `pixel`: 일반 픽셀 레이어
- `type`: 텍스트 레이어
- `shape`: 벡터 쉐이프
- `smartobject`: 스마트 오브젝트
- `adjustment`: 조정 레이어 (Curves, Levels 등)

### 텍스트 레이어 파싱
```python
if layer.kind == 'type':
    text_data = layer.engine_dict
    # text_data['Editor']['Text'] 에서 실제 문자열 추출
```

### 폰트 정보 추출
```python
if layer.kind == 'type':
    # ResourceDict > FontSet 에서 폰트 이름 추출
    pass
```

## 주의사항

### 1. Composite 이미지
- `layer.composite()`는 해당 레이어만 렌더링 (효과 미적용 가능)
- 조정 레이어의 영향을 받지 않음
- 클리핑 마스크는 base layer에서만 유효

### 2. 메모리 관리
- 대용량 PSD는 lazy loading 활용
- 처리 완료된 레이어는 즉시 해제

### 3. 에러 처리
```python
try:
    composite = layer.composite()
except Exception as e:
    logging.warning(f"Layer {layer.name} render failed: {e}")
    composite = None
```

## 휴리스틱 태깅 규칙

### 크기 기반
- `area > 90% canvas`: Background
- `area < 5% canvas`: Detail/Prop

### 위치 기반
- `center`: Main Object
- `corners`: UI/Decoration

### 알파 분포
- 점진적 투명도: FX/Glow
- 경계가 뚜렷함: Solid Object
