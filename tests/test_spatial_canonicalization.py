from backend.db.sqlite_client import SQLiteDB


def test_canonical_object_name_normalizes_plural_and_common_synonyms():
    assert SQLiteDB._canonical_object_name("shelves") == "shelf"
    assert SQLiteDB._canonical_object_name("cupboard") == "cabinet"
    assert SQLiteDB._canonical_object_name("bottles") == "bottle"


def test_korean_object_name_uses_known_dictionary_when_vlm_translation_is_bad():
    assert SQLiteDB._canonical_ko_name("shelf", "가까이") == "선반"
    assert SQLiteDB._canonical_ko_name("cabinet", "장롱") == "수납장"
