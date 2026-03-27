# 검색 아키텍처 정의서

## 핵심 원칙

**자연어 해석**과 **검색 실행**은 완전히 분리된다.

```
사용자 문장 → [해석기] → 검색 조건 → [검색 엔진] → 결과
```

- 해석기: 문장의 의도를 구조화된 조건으로 변환 (Codex/LLM/규칙)
- 검색 엔진: 조건을 받아서 DB에서 결과를 찾음 (FTS/VV/MV)
- 해석기는 DB를 모름. 검색 엔진은 자연어를 모름.

## 검색 조건 스키마

해석기의 출력 포맷 (모든 백엔드 공통):

```json
{
  "scope": {
    "folder": "세일러문",
    "image_type": null,
    "format": null
  },
  "find": {
    "description": "river scene with water visible",
    "keywords": ["강", "river", "water"]
  },
  "exclude": {
    "description": "",
    "keywords": []
  }
}
```

| 필드 | 역할 | 검색 엔진에서의 처리 |
|------|------|-------------------|
| `scope` | 어디서 찾을지 (범위 제한) | SQL WHERE → file_id 집합 |
| `find` | 뭘 찾을지 (검색 의도) | VV/MV 벡터 유사도 + FTS 키워드 |
| `exclude` | 뭘 제외할지 | 네거티브 필터 |

## 해석기 3단 폴백

```
Codex CLI (GPT-5.3) — 가장 정확, ~10초, 네트워크 필요
  ↓ 실패 시
MLX (Qwen3.5-4B) — 로컬, ~2초, 한국어 부분적
  ↓ 실패 시
규칙 기반 — 즉시, 조사 제거 + 번역
```

어떤 해석기를 사용하든 **출력 포맷은 동일**해야 함.

## 검색 엔진 실행

### scope가 있을 때 (Plan Search)

```
scope.folder="세일러문"
  → SQL: SELECT id FROM files WHERE folder_path LIKE '%세일러문%'
  → 141 file_ids

find.description="river scene"
  → 141건의 MV 벡터 로드 → cosine similarity → 순위
  → VV도 같은 방식 (선택적)

결과: 세일러문 폴더 내 강 관련 이미지 순
```

### scope가 없을 때 (Triaxis Search)

```
find.description="night city background"
  → VV: SigLIP2 전체 DB 시각 유사도
  → MV: Qwen3-Embedding 전체 DB 의미 유사도
  → FTS: BM25 키워드 매칭
  → RRF 병합

결과: 전체 DB에서 밤 도시 배경
```

## 예시

| 사용자 입력 | scope | find | exclude |
|-----------|-------|------|---------|
| 세일러문 중에서 강이 보이는거 | folder=세일러문 | river scene | - |
| 밤 도시 배경 | (없음) | night city background | - |
| 캐릭터인데 사람 없는거 | type=character | character illustration | person, human |
| 마캬베리즈무 실내소품 어두운것 | folder=마캬베리즈무/실내소품 | dark interior | - |
| PSD 파일만 찾아줘 | format=PSD | (없음, 전체) | - |

## 수정 필요 사항

현재 코드에서 이 스키마를 통일해야 함:
- `query_decomposer.py`: Codex/MLX/fallback 모두 같은 포맷 출력
- `sqlite_search.py`: `plan_search()`가 이 포맷을 받아 실행
- 기존 `folder_filter`, `fts_keywords`, `vector_query` 등 개별 필드를 이 스키마로 통합
