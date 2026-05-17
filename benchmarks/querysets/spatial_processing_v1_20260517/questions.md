# Spatial Processing QuerySet V1

정규화된 공간 관계와 깊이 레이어 검색을 검증하기 위한 최소 질문셋입니다.

## spatial_relation_001

컵이 테이블 위에 있는 이미지

- type: spatial_relation
- must: 컵, 테이블, 위
- soft: on

## spatial_depth_001

전경에 테이블이 있는 이미지

- type: spatial_depth
- must: 전경, 테이블
- soft: foreground
