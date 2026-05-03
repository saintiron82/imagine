import pytest

from tools.bench_precision import (
    _is_benchmark_tag_candidate,
    _is_known_visible_tag,
    _is_noisy_benchmark_tag,
)


@pytest.mark.parametrize(
    "tag",
    [
        "imagine_dl_m37a152c",
        "imagine_worker_yogxs3e7",
        "thumb_33767.png",
        "COS012.jpg",
        "A-46",
        "platform_30",
        "number_6",
    ],
)
def test_noisy_benchmark_tags_are_excluded(tag):
    assert _is_noisy_benchmark_tag(tag)
    assert not _is_benchmark_tag_candidate(tag)


@pytest.mark.parametrize("tag", ["window", "vase", "stone_background", "modern_architecture"])
def test_user_visible_tags_are_candidates(tag):
    assert not _is_noisy_benchmark_tag(tag)
    assert _is_benchmark_tag_candidate(tag)


@pytest.mark.parametrize("tag", ["anime", "anime character", "digital", "warm_palette"])
def test_abstract_skip_tags_are_not_candidates(tag):
    assert not _is_benchmark_tag_candidate(tag)


@pytest.mark.parametrize("tag", ["window", "city street", "character sketch"])
def test_known_visible_tags_are_detected(tag):
    assert _is_known_visible_tag(tag)


@pytest.mark.parametrize("tag", ["still_life", "rustic", "blue_tone", "papers", "stone_background"])
def test_non_catalogued_tags_are_not_known_visible_tags(tag):
    assert not _is_known_visible_tag(tag)
