"""agentcli integration in QueryDecomposer._generate_codex."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_agentcli_singleton():
    """Reset the class-level agentcli singleton between tests."""
    from backend.search.query_decomposer import QueryDecomposer
    QueryDecomposer._AGENTCLI_CHECKED = False
    QueryDecomposer._AGENTCLI_CLIENT = None
    yield
    QueryDecomposer._AGENTCLI_CHECKED = False
    QueryDecomposer._AGENTCLI_CLIENT = None


def test_codex_prompt_contains_documented_fields():
    """Decomposer prompt asks for the exact JSON keys the parser expects."""
    from backend.search.query_decomposer import QueryDecomposer

    d = QueryDecomposer(use_codex=False)
    prompt = d._build_codex_prompt("세일러문 중에서 강이 보이는거")
    assert "pre_filter" in prompt
    assert '"folder"' in prompt
    assert '"search"' in prompt
    assert "fallback_keywords" in prompt
    # Query must end up in the prompt
    assert "세일러문" in prompt


def test_codex_uses_agentcli_when_available(monkeypatch):
    """If agentcli is installed, _generate_codex uses it and parses JSON."""
    from backend.search.query_decomposer import QueryDecomposer

    fake_payload = (
        '{"pre_filter": {"folder": "세일러문", "image_type": null, "format": null}, '
        '"search": {"query": "river scene", "mode": "semantic"}, '
        '"fallback_keywords": ["river", "강"]}'
    )

    fake_client = SimpleNamespace(
        chat=lambda **kwargs: SimpleNamespace(content=fake_payload),
    )
    monkeypatch.setattr(
        QueryDecomposer, "_AGENTCLI_CHECKED", True, raising=False,
    )
    monkeypatch.setattr(
        QueryDecomposer, "_AGENTCLI_CLIENT", fake_client, raising=False,
    )

    d = QueryDecomposer(use_codex=True)
    out = d._generate_codex("세일러문 중에서 강이 보이는거")
    assert out is not None
    assert out.startswith("{")
    assert "세일러문" in out
    assert "river" in out


def test_codex_extracts_json_from_prose_wrapper(monkeypatch):
    """When the LLM wraps JSON in prose, we still extract the object."""
    from backend.search.query_decomposer import QueryDecomposer

    noisy = (
        "Here is the search plan you asked for:\n"
        '{"pre_filter": {"folder": "", "image_type": null, "format": null}, '
        '"search": {"query": "night city", "mode": "semantic"}, '
        '"fallback_keywords": ["night", "city"]}\n'
        "Let me know if you need anything else."
    )
    fake_client = SimpleNamespace(
        chat=lambda **kwargs: SimpleNamespace(content=noisy),
    )
    monkeypatch.setattr(QueryDecomposer, "_AGENTCLI_CHECKED", True, raising=False)
    monkeypatch.setattr(QueryDecomposer, "_AGENTCLI_CLIENT", fake_client, raising=False)

    d = QueryDecomposer(use_codex=True)
    out = d._generate_codex("밤 도시 배경")
    assert out is not None
    assert out.startswith("{") and out.endswith("}")
    assert "night city" in out
    # No prose leak
    assert "Here is" not in out
    assert "Let me know" not in out


def test_codex_handles_nested_braces():
    """Balanced-brace extractor must keep nested objects intact."""
    from backend.search.query_decomposer import QueryDecomposer
    from types import SimpleNamespace as SN

    # pre_filter is itself a nested object
    payload = (
        '{"pre_filter": {"folder": "x", "image_type": null}, '
        '"search": {"query": "y", "mode": "semantic"}, '
        '"fallback_keywords": []}'
    )

    fake_client = SN(chat=lambda **kw: SN(content=f"PREFIX {payload} SUFFIX"))
    QueryDecomposer._AGENTCLI_CHECKED = True
    QueryDecomposer._AGENTCLI_CLIENT = fake_client
    try:
        d = QueryDecomposer(use_codex=True)
        out = d._generate_codex("x y")
        # Verify the full nested object survived; extractor returns
        # the FIRST balanced top-level object (the outer one).
        assert out is not None
        import json as _j
        parsed = _j.loads(out)
        assert parsed["pre_filter"]["folder"] == "x"
        assert parsed["search"]["query"] == "y"
    finally:
        QueryDecomposer._AGENTCLI_CHECKED = False
        QueryDecomposer._AGENTCLI_CLIENT = None


def test_codex_falls_back_to_legacy_when_agentcli_raises(monkeypatch):
    """A raised exception inside agentcli must not break the caller —
    we fall through to the legacy subprocess path."""
    from backend.search.query_decomposer import QueryDecomposer

    def _boom(**kwargs):
        raise RuntimeError("agentcli misbehaving")

    fake_client = SimpleNamespace(chat=_boom)
    monkeypatch.setattr(QueryDecomposer, "_AGENTCLI_CHECKED", True, raising=False)
    monkeypatch.setattr(QueryDecomposer, "_AGENTCLI_CLIENT", fake_client, raising=False)

    called = {"n": 0}

    def fake_legacy(self, query):
        called["n"] += 1
        return '{"pre_filter":{"folder":""},"search":{"query":"x","mode":"semantic"},"fallback_keywords":[]}'

    monkeypatch.setattr(QueryDecomposer, "_generate_codex_legacy", fake_legacy)

    d = QueryDecomposer(use_codex=True)
    out = d._generate_codex("아무거나")
    assert called["n"] == 1
    assert out is not None
    assert out.startswith("{")


def test_legacy_path_is_used_when_agentcli_not_installed(monkeypatch):
    """If agentcli import fails, the lazy init returns None and legacy fires."""
    from backend.search.query_decomposer import QueryDecomposer

    # Force the lazy-init to think agentcli is missing.
    monkeypatch.setattr(QueryDecomposer, "_AGENTCLI_CHECKED", True, raising=False)
    monkeypatch.setattr(QueryDecomposer, "_AGENTCLI_CLIENT", None, raising=False)

    called = {"n": 0}

    def fake_legacy(self, query):
        called["n"] += 1
        return None  # legacy decides nothing matched — None is a valid result

    monkeypatch.setattr(QueryDecomposer, "_generate_codex_legacy", fake_legacy)

    d = QueryDecomposer(use_codex=True)
    out = d._generate_codex("foo")
    assert called["n"] == 1
    assert out is None
