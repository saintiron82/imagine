import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISION_ROOT = ROOT / "backend" / "vision"


def load_vision_modules():
    backend_pkg = types.ModuleType("backend")
    backend_pkg.__path__ = [str(ROOT / "backend")]
    vision_pkg = types.ModuleType("backend.vision")
    vision_pkg.__path__ = [str(VISION_ROOT)]
    sys.modules.setdefault("backend", backend_pkg)
    sys.modules.setdefault("backend.vision", vision_pkg)

    loaded = {}
    for name in ("schemas", "repair"):
        module_name = f"backend.vision.{name}"
        spec = importlib.util.spec_from_file_location(module_name, VISION_ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def test_parse_structured_output_attaches_diagnostics_for_direct_json():
    modules = load_vision_modules()

    parsed = modules["repair"].parse_structured_output(
        '{"caption":"x","objects":[]}',
        modules["schemas"].get_schema("background"),
        image_type="background",
        include_diagnostics=True,
    )

    assert parsed["_parse_diagnostics"]["status"] == "direct"
    assert parsed["_parse_diagnostics"]["repaired"] is False


def test_parse_structured_output_attaches_diagnostics_for_fallback_json():
    modules = load_vision_modules()

    parsed = modules["repair"].parse_structured_output(
        '{"caption":"x","objects":[{"name":"moon","locations":["right"]}],',
        modules["schemas"].get_schema("background"),
        image_type="background",
        include_diagnostics=True,
    )

    assert parsed["_parse_diagnostics"]["status"] in {"repaired", "fallback"}
    assert parsed["_parse_diagnostics"]["repaired"] is True
