from backend.api_search import format_result


def test_format_result_surfaces_spatial_relations_and_depth_layers():
    result = {
        "id": 1,
        "file_path": "/tmp/example.png",
        "metadata": {},
        "spatial_objects": [{"name": "cup"}],
        "spatial_relations": [{"subject": "cup", "relation": "on", "object": "table"}],
        "depth_layers": [{"name": "table", "layer": "foreground"}],
        "spatial_processing_quality": {"objects_status": "ok"},
        "metadata_reliability_score": 0.92,
        "metadata_quality_source": "profile_inferred",
        "metadata_quality_basis": ["global"],
    }

    formatted = format_result(result, skip_fs=True)

    assert formatted["spatial_objects"] == [{"name": "cup"}]
    assert formatted["spatial_relations"] == [
        {"subject": "cup", "relation": "on", "object": "table"}
    ]
    assert formatted["depth_layers"] == [{"name": "table", "layer": "foreground"}]
    assert formatted["spatial_processing_quality"] == {"objects_status": "ok"}
    assert formatted["metadata_reliability_score"] == 0.92
    assert formatted["metadata_quality_source"] == "profile_inferred"
    assert formatted["metadata_quality_basis"] == ["global"]
