"""Phase 1: Firebase registry must publish connection mode metadata.

The server registers itself in Firestore so external clients can discover
its connection modes (direct LAN, manual external, relay session). The
payload must carry `connect_mode` plus URL/relay slots so the frontend
can choose a candidate without guessing.
"""

from __future__ import annotations

import json


def _capture_payload(monkeypatch) -> dict:
    """Stub urlopen so register_group writes are captured locally."""
    from backend.server import firebase_registry

    captured: dict = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b""

    def fake_urlopen(req, timeout):  # noqa: ANN001 - urllib signature
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        if req.data:
            captured["body"] = req.data.decode("utf-8")
        return _Resp()

    monkeypatch.setattr(firebase_registry.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(firebase_registry, "_get_lan_ip", lambda: "192.168.0.8")
    monkeypatch.setattr(firebase_registry, "_get_public_ip", lambda: "")
    return captured


def test_registry_payload_contains_connection_modes(monkeypatch):
    from backend.server import firebase_registry

    captured = _capture_payload(monkeypatch)
    assert firebase_registry.register_group("studio", 8000) is True

    body = captured["body"]
    for field in (
        "connect_mode",
        "lan_url",
        "external_url",
        "tunnel_url",
        "relay_endpoint",
        "relay_online",
        "reachable_status",
        "updated_at",
    ):
        assert f'"{field}"' in body, f"missing {field} in registry payload: {body}"


def test_registry_defaults_to_direct_lan_when_only_lan_known(monkeypatch):
    from backend.server import firebase_registry

    captured = _capture_payload(monkeypatch)
    assert firebase_registry.register_group("studio", 8000) is True

    payload = json.loads(captured["body"])
    fields = payload["fields"]

    assert fields["connect_mode"]["stringValue"] == "direct_lan"
    assert fields["lan_url"]["stringValue"] == "http://192.168.0.8:8000"
    assert fields["external_url"]["stringValue"] == ""
    assert fields["tunnel_url"]["stringValue"] == ""
    assert fields["relay_endpoint"]["stringValue"] == ""
    assert fields["relay_online"]["booleanValue"] is False
    assert fields["reachable_status"]["stringValue"] == "unknown"


def test_registry_overrides_via_arguments(monkeypatch):
    from backend.server import firebase_registry

    captured = _capture_payload(monkeypatch)
    assert (
        firebase_registry.register_group(
            "studio",
            8000,
            connect_mode="relay_session",
            external_url="https://imagine.example.com",
            tunnel_url="https://abcd.trycloudflare.com",
            relay_endpoint="wss://relay.example.com",
            relay_online=True,
            reachable_status="verified",
        )
        is True
    )

    payload = json.loads(captured["body"])
    fields = payload["fields"]
    assert fields["connect_mode"]["stringValue"] == "relay_session"
    assert fields["external_url"]["stringValue"] == "https://imagine.example.com"
    assert fields["tunnel_url"]["stringValue"] == "https://abcd.trycloudflare.com"
    assert fields["relay_endpoint"]["stringValue"] == "wss://relay.example.com"
    assert fields["relay_online"]["booleanValue"] is True
    assert fields["reachable_status"]["stringValue"] == "verified"
