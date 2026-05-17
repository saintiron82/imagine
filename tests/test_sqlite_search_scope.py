import types

from backend.search.sqlite_search import SqliteVectorSearch


def test_fts_mode_forwards_file_ids_to_fts_search():
    searcher = object.__new__(SqliteVectorSearch)
    calls = []

    def fake_fts_search(self, keywords, top_k, exclude_keywords=None, file_ids=None):
        calls.append((keywords, top_k, exclude_keywords, file_ids))
        return []

    searcher.fts_search = types.MethodType(fake_fts_search, searcher)

    searcher.search("오른쪽 위 구름", mode="fts", top_k=5, file_ids={1, 2, 3})

    assert calls == [(["오른쪽 위 구름"], 5, None, {1, 2, 3})]
