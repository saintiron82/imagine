"""Auto-Weighted RRF strategy for 3-axis search (V + T + F).

Selects per-axis weights based on query_type from QueryDecomposer.
Falls back to manual_weights from config.yaml when auto_weight is disabled.
"""

import logging
from typing import Dict, List

from backend.utils.config import get_config

logger = logging.getLogger(__name__)

# Weight presets per query type (must sum to 1.0)
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "visual":   {"visual": 0.50, "text_vec": 0.30, "fts": 0.20},
    "keyword":  {"visual": 0.20, "text_vec": 0.30, "fts": 0.50},
    "semantic": {"visual": 0.20, "text_vec": 0.50, "fts": 0.30},
    "balanced": {"visual": 0.34, "text_vec": 0.33, "fts": 0.33},
}


def get_weights(query_type: str, active_axes: List[str]) -> Dict[str, float]:
    """
    Return per-axis weights based on query_type and active axes.

    Args:
        query_type: One of "visual", "keyword", "semantic", "balanced".
        active_axes: List of active axis names (e.g. ["visual", "text_vec", "fts"]).

    Returns:
        Dict mapping axis name -> weight. Weights sum to 1.0.
    """
    cfg = get_config()
    auto = cfg.get("search.rrf.auto_weight", True)

    if auto:
        base = WEIGHT_PRESETS.get(query_type, WEIGHT_PRESETS["balanced"]).copy()
    else:
        manual = cfg.get("search.rrf.manual_weights", {})
        base = {
            "visual": manual.get("visual", 0.34),
            "text_vec": manual.get("text_vec", manual.get("text", 0.33)),
            "fts": manual.get("fts", 0.33),
        }

    weights = _redistribute(base, active_axes)
    logger.debug(f"RRF weights: type={query_type}, auto={auto}, active={active_axes}, w={weights}")
    return weights


def _redistribute(weights: Dict[str, float], active_axes: List[str]) -> Dict[str, float]:
    """Redistribute inactive axis weights proportionally to active axes."""
    active_weight = sum(weights.get(a, 0) for a in active_axes)

    if active_weight <= 0:
        # All axes inactive — equal split among whatever is active
        n = len(active_axes) if active_axes else 1
        return {a: 1.0 / n for a in active_axes}

    # Scale active axes so they sum to 1.0
    return {a: weights.get(a, 0) / active_weight for a in active_axes}
