from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.server.routers.server_init import _require_init_allowed


def _request(host, setup_token=None):
    headers = {}
    if setup_token:
        headers["x-imagine-setup-token"] = setup_token
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


def test_init_allows_loopback_without_setup_token(monkeypatch):
    monkeypatch.delenv("IMAGINE_SETUP_TOKEN", raising=False)

    _require_init_allowed(_request("127.0.0.1"))
    _require_init_allowed(_request("::1"))


def test_init_rejects_lan_without_setup_token(monkeypatch):
    monkeypatch.delenv("IMAGINE_SETUP_TOKEN", raising=False)

    with pytest.raises(HTTPException) as exc:
        _require_init_allowed(_request("192.168.1.25"))

    assert exc.value.status_code == 403


def test_init_allows_lan_with_matching_setup_token(monkeypatch):
    monkeypatch.setenv("IMAGINE_SETUP_TOKEN", "setup-secret")

    _require_init_allowed(_request("192.168.1.25", setup_token="setup-secret"))


def test_init_rejects_lan_with_wrong_setup_token(monkeypatch):
    monkeypatch.setenv("IMAGINE_SETUP_TOKEN", "setup-secret")

    with pytest.raises(HTTPException) as exc:
        _require_init_allowed(_request("192.168.1.25", setup_token="wrong"))

    assert exc.value.status_code == 403
