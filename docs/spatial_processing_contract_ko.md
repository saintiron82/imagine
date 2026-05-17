# 공간 정보 가공 계약

## 단계

1. 원천 추출: VLM raw output
2. 파싱 원본: `files.structured_meta`
3. 정규화 저장: `file_objects`, `file_spatial_relations`, `file_depth_layers`
4. 검색 가공: `files_fts.spatial`
5. 활용: API search results and file detail

## 기본 필드

- `objects`
- `relations`
- `depth_layers`
- `spatial_processing_quality`
- `spatial_schema_version`
- `vlm_raw_outputs`

## 운영 규칙

- raw output은 변경하지 않는다.
- `structured_meta`는 현재 파서 기준의 파싱 결과다.
- 정규화 테이블은 검색과 UI를 위한 파생 데이터다.
- FTS는 재생성 가능한 파생 인덱스다.
- 품질 상태가 `failed` 또는 `partial`이면 재처리 후보가 된다.
