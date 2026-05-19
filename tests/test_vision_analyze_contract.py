from PIL import Image

from backend.vision.ollama_adapter import OllamaVisionAdapter


def test_ollama_analyze_delegates_to_spatial_two_stage_pipeline():
    adapter = object.__new__(OllamaVisionAdapter)
    calls = []

    def classify_and_analyze(image, *, context=None, domain=None):
        calls.append({"context": context, "domain": domain, "mode": image.mode})
        return {"caption": "spatial", "tags": ["spatial"], "spatial_schema_version": 2}

    adapter.classify_and_analyze = classify_and_analyze

    result = adapter.analyze(Image.new("RGB", (2, 2)), {"file_name": "asset.png"})

    assert result["spatial_schema_version"] == 2
    assert calls == [
        {"context": {"file_name": "asset.png"}, "domain": None, "mode": "RGB"}
    ]
