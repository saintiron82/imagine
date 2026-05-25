import pytest


def test_headless_cli_requires_access_token(monkeypatch):
    from backend.worker.cli import load_headless_config

    monkeypatch.setenv("IMAGINE_SERVER_URL", "http://server")
    monkeypatch.delenv("IMAGINE_WORKER_ACCESS_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc:
        load_headless_config([])

    assert exc.value.code == 2


def test_headless_cli_loads_env_config(monkeypatch):
    from backend.worker.cli import load_headless_config

    monkeypatch.setenv("IMAGINE_SERVER_URL", "http://server")
    monkeypatch.setenv("IMAGINE_WORKER_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("IMAGINE_WORKER_REFRESH_TOKEN", "refresh-token")
    monkeypatch.setenv("IMAGINE_WORKER_NAME", "gpu-node")
    monkeypatch.setenv("IMAGINE_WORKER_LAUNCHER", "cloud")

    cfg = load_headless_config([])

    assert cfg.server_url == "http://server"
    assert cfg.access_token == "access-token"
    assert cfg.refresh_token == "refresh-token"
    assert cfg.worker_name == "gpu-node"
    assert cfg.origin == "headless"
    assert cfg.launcher == "cloud"
