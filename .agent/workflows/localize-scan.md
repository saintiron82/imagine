---
description: 프론트엔드 하드코딩 문자열 탐색 및 리포트
---
# /localize-scan

프론트엔드 JSX 파일에서 하드코딩된 UI 문자열을 탐색하고 리포트를 생성합니다.

## Steps

// turbo
1. 번역 파일 키 목록 로드
```powershell
# 현재 등록된 키 수 확인
python -c "import json; en=json.load(open('frontend/src/i18n/locales/en-US.json','r',encoding='utf-8')); ko=json.load(open('frontend/src/i18n/locales/ko-KR.json','r',encoding='utf-8')); print(f'en-US: {len(en)} keys, ko-KR: {len(ko)} keys'); missing_en=set(ko)-set(en); missing_ko=set(en)-set(ko); print(f'Missing in en-US: {missing_en or \"none\"}'); print(f'Missing in ko-KR: {missing_ko or \"none\"}')"
```

2. JSX 파일 스캔
   - `frontend/src/App.jsx` 및 `frontend/src/components/*.jsx` 읽기
   - 하드코딩된 문자열 패턴 탐지:
     - `>텍스트<` (JSX 텍스트 노드)
     - `placeholder="텍스트"`, `title="텍스트"` (속성)
     - 삼항 연산자 내 문자열 리터럴
   - `t('...')` 호출이 아닌 UI 문자열 식별

3. 미사용 키 탐지
   - 번역 파일의 모든 키에 대해 코드 내 `t('key')` 참조 검색
   - 참조되지 않는 키 리스트 생성

4. 리포트 출력
   - 파일별 하드코딩 문자열 목록
   - 누락 키 목록 (en-US vs ko-KR)
   - 미사용 키 목록
   - 권장 조치 사항
