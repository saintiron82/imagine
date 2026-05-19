from PIL import Image

from backend.pipeline.phase_runner import PhaseRunner
from backend.pipeline.protocols import PhaseItem


class _SpatialOnlyAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, *_args, **_kwargs):
        raise AssertionError("legacy analyze() must not be used by PhaseRunner")

    def classify_and_analyze(self, image, *, context=None, domain=None):
        self.calls.append({"context": context, "domain": domain, "mode": image.mode})
        return {
            "caption": "red chair beside a window",
            "tags": ["chair", "window"],
            "objects": [{"name": "chair", "primary_location": "center"}],
            "relations": [{"subject": "chair", "relation": "near", "object": "window"}],
            "depth_layers": [{"name": "chair", "layer": "foreground"}],
            "spatial_schema_version": 2,
        }


class _Models:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.unloaded = False

    def get_vlm(self):
        return self.analyzer

    def unload_vlm(self):
        self.unloaded = True


class _Storage:
    def __init__(self):
        self.saved = []

    def save_vision(self, item, result):
        self.saved.append((item.file_id, result))


def test_phase_runner_vision_uses_spatial_two_stage_contract(monkeypatch, tmp_path):
    image_path = tmp_path / "asset.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path)
    analyzer = _SpatialOnlyAnalyzer()
    models = _Models(analyzer)
    storage = _Storage()
    monkeypatch.setattr(PhaseRunner, "_get_active_domain", lambda _self: "active-domain")

    runner = PhaseRunner(models=models, storage=storage)
    items = runner.run_vision([
        PhaseItem(
            file_id=42,
            file_path=str(image_path),
            thumb_path=str(image_path),
            mc_raw={"file_name": "asset.png"},
        )
    ])

    assert items[0].vision_result["spatial_schema_version"] == 2
    assert storage.saved[0][1]["objects"][0]["name"] == "chair"
    assert analyzer.calls == [
        {
            "context": {"file_name": "asset.png"},
            "domain": "active-domain",
            "mode": "RGB",
        }
    ]
    assert models.unloaded is True
