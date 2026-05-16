from backend.search.query_decomposer import QueryDecomposer


def test_korean_spatial_query_particles_keep_object_and_location_terms():
    decomposer = QueryDecomposer(use_codex=False)

    assert decomposer._extract_ko_keywords("오른쪽에 달이 있다") == ["오른쪽", "달"]
    assert decomposer._extract_ko_keywords("달 오른쪽") == ["달", "오른쪽"]
