"""
IR evaluation metrics for image search benchmark.
Supports graded relevance (0/1/2).
"""

import math
from typing import Dict, List, Optional


def ndcg_at_k(
    ranked_ids: List[str],
    relevance: Dict[str, int],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_ids: List[str],
    relevant: set,
    k: int = 50,
) -> float:
    """Recall @ K - fraction of relevant items found in top K."""
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return found / len(relevant)


def mrr_at_k(
    ranked_ids: List[str],
    relevant: set,
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank @ K."""
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def p_at_k(
    ranked_ids: List[str],
    relevant: set,
    k: int = 10,
) -> float:
    """Precision @ K."""
    if k == 0:
        return 0.0
    found = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return found / k


def compute_query_metrics(
    ranked_ids: List[str],
    relevance: Dict[str, int],
    latency_ms: int = 0,
) -> dict:
    """Compute all metrics for a single query."""
    relevant_set = {doc_id for doc_id, rel in relevance.items() if rel > 0}

    return {
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, k=10),
        "recall@50": recall_at_k(ranked_ids, relevant_set, k=50),
        "mrr@10": mrr_at_k(ranked_ids, relevant_set, k=10),
        "p@10": p_at_k(ranked_ids, relevant_set, k=10),
        "latency_ms": latency_ms,
    }


def aggregate_metrics(
    query_metrics: List[dict],
) -> dict:
    """Aggregate per-query metrics into summary statistics."""
    if not query_metrics:
        return {}

    n = len(query_metrics)
    metric_names = ["ndcg@10", "recall@50", "mrr@10", "p@10"]
    latencies = [m["latency_ms"] for m in query_metrics if m.get("latency_ms")]

    result = {"n_queries": n}

    for name in metric_names:
        values = [m[name] for m in query_metrics]
        result[name] = sum(values) / n

    if latencies:
        latencies_sorted = sorted(latencies)
        result["mean_latency_ms"] = sum(latencies) / len(latencies)
        idx_95 = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
        result["p95_latency_ms"] = latencies_sorted[idx_95]

    result["error_rate"] = sum(
        1 for m in query_metrics if m.get("error")
    ) / n

    # per-query scores (for bootstrap)
    result["_per_query"] = {
        name: [m[name] for m in query_metrics]
        for name in metric_names
    }

    return result
