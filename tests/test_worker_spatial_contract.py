import sqlite3
import sys
import types

from PIL import Image

sys.modules.setdefault(
    "jwt",
    types.SimpleNamespace(
        ExpiredSignatureError=Exception,
        InvalidTokenError=Exception,
        decode=lambda *args, **kwargs: {},
        encode=lambda *args, **kwargs: "",
    ),
)

from backend.server.routers.analysis import _save_vision_fields_for_file
from backend.worker.transport import LocalTransport
from backend.worker.worker_daemon import WorkerDaemon, _vision_result_to_fields


def _spatial_vision_result():
    return {
        "caption": "red chair beside a window",
        "tags": ["chair", "window"],
        "image_type": "photo",
        "caption_model": "mlx-community/Qwen3.5-9B",
        "objects": [
            {
                "name": "chair",
                "canonical_name": "chair",
                "locations": ["center"],
                "primary_location": "center",
            }
        ],
        "relations": [
            {
                "subject": "chair",
                "relation": "beside",
                "object": "window",
                "confidence": "high",
            }
        ],
        "depth_layers": [
            {
                "layer": "foreground",
                "objects": ["chair"],
                "confidence": "high",
            }
        ],
        "_vlm_raw": '{"caption":"red chair beside a window"}',
        "_vlm_provenance": {
            "stage": "stage2",
            "adapter": "MLXVisionAdapter",
            "model": "mlx-community/Qwen3.5-9B",
            "prompt_version": "spatial_v2",
        },
        "_parse_diagnostics": {"status": "direct", "repaired": False},
    }


def test_worker_vision_result_to_fields_preserves_spatial_payload():
    fields = _vision_result_to_fields(_spatial_vision_result())

    assert fields["mc_caption"] == "red chair beside a window"
    assert fields["ai_tags"] == ["chair", "window"]
    assert fields["image_type"] == "photo"
    assert fields["caption_model"] == "mlx-community/Qwen3.5-9B"

    structured_meta = fields["structured_meta"]
    assert structured_meta["caption"] == "red chair beside a window"
    assert structured_meta["objects"][0]["canonical_name"] == "chair"
    assert structured_meta["relations"][0]["relation"] == "beside"
    assert structured_meta["depth_layers"][0]["layer"] == "foreground"
    assert structured_meta["_vlm_raw"] == '{"caption":"red chair beside a window"}'
    assert structured_meta["_vlm_provenance"]["prompt_version"] == "spatial_v2"
    assert structured_meta["_parse_diagnostics"]["status"] == "direct"


class RecordingDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE files(id INTEGER PRIMARY KEY, file_path TEXT, structured_meta TEXT)"
        )
        self.conn.execute(
            "INSERT INTO files(id, file_path, structured_meta) VALUES (42, '/asset.png', '{}')"
        )
        self.conn.commit()
        self.updated = []

    def update_vision_fields(self, file_path, fields):
        self.updated.append((file_path, fields))
        return True


def test_local_transport_routes_vision_through_spatial_storage_contract():
    db = RecordingDB()
    transport = LocalTransport(db, scheduler=None, manager=None)

    assert transport.save_vision(42, {"structured_meta": _spatial_vision_result()}) is True

    assert db.updated == [
        ("/asset.png", {"structured_meta": _spatial_vision_result()})
    ]


def test_server_vision_endpoint_helper_routes_through_spatial_storage_contract():
    db = RecordingDB()

    assert _save_vision_fields_for_file(
        db,
        42,
        {"structured_meta": _spatial_vision_result(), "caption_model": "mlx"},
    ) is True

    assert db.updated == [
        (
            "/asset.png",
            {"structured_meta": _spatial_vision_result(), "caption_model": "mlx"},
        )
    ]


def test_worker_run_vision_uses_spatial_two_stage_analyzer(monkeypatch, tmp_path):
    class SpatialOnlyAnalyzer:
        def analyze(self, *_args, **_kwargs):
            raise AssertionError("legacy analyze() must not be used for worker MC")

        def classify_and_analyze(self, image, *, context=None, domain=None):
            assert context["file_name"] == "asset.png"
            assert domain == "active-domain"
            assert image.mode == "RGB"
            return _spatial_vision_result()

    image_path = tmp_path / "asset.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(image_path)

    monkeypatch.setattr(
        "backend.vision.vision_factory.get_vision_analyzer",
        lambda: SpatialOnlyAnalyzer(),
    )
    monkeypatch.setattr(
        "backend.vision.domain_loader.get_active_domain",
        lambda: "active-domain",
    )

    fields = WorkerDaemon._run_vision(
        object.__new__(WorkerDaemon),
        image_path,
        str(image_path),
        meta=None,
        mc_raw_override={"file_name": "asset.png"},
    )

    assert fields["mc_caption"] == "red chair beside a window"
    assert fields["structured_meta"]["_vlm_provenance"]["prompt_version"] == "spatial_v2"
