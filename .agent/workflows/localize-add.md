---
description: 새 번역 키 추가 (양쪽 언어 파일 동시 업데이트)
---
# /localize-add

새로운 번역 키를 en-US.json과 ko-KR.json에 동시 추가합니다.

## Steps

1. 키 정보 수집
   - 키 이름 (네이밍 컨벤션 준수: `prefix.name`)
   - 영어 값
   - 한국어 값
   - 보간 파라미터 확인 (`{param}` 구문)

2. 중복 확인
   - 기존 키와 충돌 여부 확인
   - 유사 키 제안 (이미 존재하는 경우)

// turbo
3. 양쪽 파일 업데이트
   - `frontend/src/i18n/locales/en-US.json`에 영어 값 추가
   - `frontend/src/i18n/locales/ko-KR.json`에 한국어 값 추가
   - 키는 알파벳 순서로 삽입 (prefix 그룹 내)

4. 검증
   - 양쪽 파일의 키 개수 일치 확인
   - JSON 문법 유효성 확인
   - 보간 파라미터 일치 확인

## Example

```
키: action.export
영어: Export
한국어: 내보내기

키: status.items_found (보간)
영어: {count} items found
한국어: {count}개 항목 발견
```
