"""Scope evidence validation — LLM-proposed image_type/format must be
backed by the query text, otherwise it silently collapses the scope
(e.g. hallucinated image_type='ui_element' for '도시 낮에서 거울과 거리
있는 이미지' shrank the scope to 9 files and dropped every GT)."""

from backend.search.query_decomposer import validate_scope_evidence


def _scope(folder="", image_type=None, fmt=None):
    return {"folder": folder, "image_type": image_type, "format": fmt}


def test_hallucinated_image_type_is_dropped():
    scope, dropped = validate_scope_evidence(
        _scope(image_type="ui_element"),
        "도시 낮에서 거울과 거리 있는 이미지",
    )
    assert scope["image_type"] is None
    assert dropped == ["image_type=ui_element"]


def test_evidenced_image_type_korean_is_kept():
    scope, dropped = validate_scope_evidence(
        _scope(image_type="background"),
        "중세 거실 배경 이미지",
    )
    assert scope["image_type"] == "background"
    assert dropped == []


def test_evidenced_image_type_english_is_kept():
    scope, dropped = validate_scope_evidence(
        _scope(image_type="character"),
        "red dress character standing",
    )
    assert scope["image_type"] == "character"
    assert dropped == []


def test_unknown_image_type_value_is_dropped():
    scope, dropped = validate_scope_evidence(
        _scope(image_type="scenery"),  # not in vocabulary
        "아름다운 풍경 scenery 이미지",
    )
    assert scope["image_type"] is None
    assert dropped == ["image_type=scenery"]


def test_format_without_mention_is_dropped():
    scope, dropped = validate_scope_evidence(
        _scope(fmt="PSD"),
        "밤하늘 배경",
    )
    assert scope["format"] is None
    assert dropped == ["format=PSD"]


def test_format_with_mention_is_kept():
    scope, dropped = validate_scope_evidence(
        _scope(fmt="PSD"),
        "밤하늘 배경 PSD 파일",
    )
    assert scope["format"] == "PSD"
    assert dropped == []


def test_jpeg_alias_counts_as_jpg_evidence():
    scope, dropped = validate_scope_evidence(
        _scope(fmt="jpg"),
        "jpeg 사진 원본",
    )
    assert scope["format"] == "jpg"
    assert dropped == []


def test_folder_scope_is_never_touched():
    scope, dropped = validate_scope_evidence(
        _scope(folder="크랑베르무", image_type="ui_element"),
        "크랑베르무에서 호수 이미지",
    )
    assert scope["folder"] == "크랑베르무"
    assert scope["image_type"] is None
    assert dropped == ["image_type=ui_element"]
