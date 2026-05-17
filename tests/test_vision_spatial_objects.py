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
    for name in ("schemas", "prompts", "repair"):
        module_name = f"backend.vision.{name}"
        spec = importlib.util.spec_from_file_location(module_name, VISION_ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def test_stage2_schema_includes_spatial_objects():
    schemas = load_vision_modules()["schemas"]

    schema = schemas.get_schema("background")

    assert "objects" in schema
    assert "depth_layers" in schema
    assert "relations" in schema
    assert "top-left" in schema["objects"]
    assert "primary_location" in schema["objects"]
    assert "locations" in schema["objects"]
    assert "foreground" in schema["depth_layers"]
    assert "behind" in schema["relations"]


def test_stage2_prompts_request_object_locations_in_full_and_concise_modes():
    prompts = load_vision_modules()["prompts"]

    full_prompt = prompts.get_stage2_prompt("background", concise=False)
    concise_prompt = prompts.get_stage2_prompt("background", concise=True)

    for prompt in (full_prompt, concise_prompt):
        assert '"objects"' in prompt
        assert '"depth_layers"' in prompt
        assert '"relations"' in prompt
        assert "Object-location extraction" in prompt
        assert "top-left, top, top-right" in prompt
        assert "Use multiple locations" in prompt
        assert "foreground, midground, background" in prompt
        assert "maximum 5" in prompt


def test_analysis_profile_prior_is_injected_into_stage1_and_stage2_prompts():
    prompts = load_vision_modules()["prompts"]
    profile = {
        "domain_id": "illustration",
        "expected_types": ["background", "effect"],
        "primary_type": "background",
        "source": "user",
    }

    stage1 = prompts.build_stage1_prompt(analysis_profile=profile)
    stage2 = prompts.get_stage2_prompt(
        "background", analysis_profile=profile, concise=False
    )
    concise = prompts.get_stage2_prompt(
        "background", analysis_profile=profile, concise=True
    )

    for prompt in (stage1, stage2, concise):
        assert "Analysis job profile" in prompt
        assert "background, effect" in prompt
        assert "primary expected type: background" in prompt
        assert "Do not override clear visual evidence" in prompt


def test_repair_fallback_returns_empty_objects_list():
    repair = load_vision_modules()["repair"]
    schemas = load_vision_modules()["schemas"]

    parsed = repair.parse_structured_output("not json", schemas.get_schema("background"))

    assert "objects" in parsed
    assert parsed["objects"] == []


def test_repair_fallback_preserves_nested_spatial_objects():
    repair = load_vision_modules()["repair"]
    schemas = load_vision_modules()["schemas"]
    raw = '''
    {
      "caption": "A moon appears on the right side.",
      "objects": [
        {
          "name": "moon",
          "ko_name": "달",
          "locations": ["right", "top-right"],
          "primary_location": "right",
          "extent": "small",
          "confidence": "high"
        }
      ],
      "tags": ["moon", "sky"],
      "broken": }
    '''

    parsed = repair.parse_structured_output(raw, schemas.get_schema("background"))

    assert parsed["objects"] == [
        {
            "name": "moon",
            "ko_name": "달",
            "locations": ["right", "top-right"],
            "primary_location": "right",
            "extent": "small",
            "confidence": "high",
        }
    ]


def test_repair_sanitizes_flat_spatial_object_tokens():
    repair = load_vision_modules()["repair"]
    schemas = load_vision_modules()["schemas"]
    raw = {
        "caption": "Forks fill the frame.",
        "objects": [
            "name",
            "forks",
            "ko_name",
            "포크",
            "locations",
            "top-left",
            "top",
            "center",
        ],
    }

    parsed = repair.parse_structured_output(
        __import__("json").dumps(raw, ensure_ascii=False),
        schemas.get_schema("item"),
    )

    assert parsed["objects"] == [
        {
            "name": "forks",
            "ko_name": "포크",
            "locations": ["top-left", "top", "center"],
            "primary_location": "top-left",
            "extent": "",
            "confidence": "low",
        }
    ]


def test_repair_preserves_depth_layers_and_relations():
    repair = load_vision_modules()["repair"]
    schemas = load_vision_modules()["schemas"]
    raw = {
        "caption": "A cup sits on a foreground table.",
        "objects": [
            {
                "name": "cup",
                "ko_name": "컵",
                "locations": ["center"],
                "primary_location": "center",
                "extent": "small",
                "confidence": "high",
            }
        ],
        "depth_layers": [
            {"name": "table", "ko_name": "테이블", "layer": "foreground", "confidence": "medium"}
        ],
        "relations": [
            {
                "subject": "cup",
                "relation": "on",
                "object": "table",
                "subject_location": "center",
                "object_location": "bottom",
                "confidence": "high",
            }
        ],
    }

    parsed = repair.parse_structured_output(
        __import__("json").dumps(raw, ensure_ascii=False),
        schemas.get_schema("item"),
    )

    assert parsed["depth_layers"] == [
        {"name": "table", "ko_name": "테이블", "layer": "foreground", "confidence": "medium"}
    ]
    assert parsed["relations"] == [
        {
            "subject": "cup",
            "relation": "on",
            "object": "table",
            "subject_location": "center",
            "object_location": "bottom",
            "confidence": "high",
        }
    ]
