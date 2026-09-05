#!/usr/bin/env python3
"""RRF 축 가중치가 실제로 검색 품질에 기여하는가.

배경: WeMM 을 도입하면 VV(visual)와 MV(text_vec)가 한 벡터로 합쳐져
backend/search/rrf.py 의 질의 유형별 가중치 손잡이를 잃는다.

    visual   유형: visual 0.50 / text_vec 0.35   (비 1.43)
    semantic 유형: visual 0.20 / text_vec 0.55   (비 0.36)

4배 차이라 설정상으로는 의미 있어 보인다. 그런데 그 튜닝이 **실제 검색 품질에
기여하는가**는 다른 문제다. 균등 가중치와 차이가 없다면 WeMM 통합으로 잃을
것도 없다.

이 스크립트는 현행 3축(VV=SigLIP2, MV, FTS)을 그대로 구성하고, 융합만
프리셋별 / 균등으로 바꿔가며 같은 쿼리셋을 돌린다. 축 구성과 후보는 완전히
동일하고 **가중치만 변수**다.

Usage:
  python tools/bench_rrf_weights.py --pool 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUERYSET = PROJECT_ROOT / "benchmarks/querysets/frozen_30_scoped_v1.json"
OUT_DIR = PROJECT_ROOT / "benchmarks/results"
CACHE_DIR = PROJECT_ROOT / "benchmarks/cache"


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def p_at_k(ranked, gt, k):
    return sum(1 for i in ranked[:k] if i in gt) / max(1, k)


def r_at_k(ranked, gt, k):
    return len(set(ranked[:k]) & gt) / max(1, len(gt))


def vv_matrix(ids, thumbs, cache_path):
    """SigLIP2(현행 pro 티어 VV) 임베딩. 캐시 재사용."""
    if cache_path.exists():
        z = np.load(cache_path)
        if list(z["ids"]) == list(ids):
            print(f"  VV 캐시 사용: {cache_path.name}")
            return z["mat"]
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    from PIL import Image
    enc = SigLIP2Encoder()
    t0 = time.perf_counter()
    mat = None
    for n, fid in enumerate(ids):
        v = None
        try:
            with Image.open(thumbs[fid]) as im:
                v = l2(enc.encode_image(im.convert("RGB")))
        except Exception:
            pass
        if mat is None:
            mat = np.zeros((len(ids), len(v) if v is not None else 1152), dtype=np.float32)
        if v is not None:
            mat[n] = v
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] {time.perf_counter()-t0:.0f}s", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(ids), mat=mat)
    print(f"  VV 인코딩 {time.perf_counter()-t0:.0f}s → 캐시 저장")
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=250)
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    args = ap.parse_args()

    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    from backend.search.scoring import rrf_merge_multi
    from backend.search.rrf import WEIGHT_PRESETS
    from backend.vector.text_embedding import get_text_embedding_provider

    queries = json.load(open(QUERYSET))["queries"]
    gt_all = {i for q in queries for i in q["gt_ids"]}

    db = SQLiteDB(args.db)
    search = SqliteVectorSearch(db=db)
    cur = db.conn.cursor()

    # 풀: 정답 우선 + 방해
    gl = sorted(gt_all)
    ph = ",".join("?" * len(gl))
    rows = cur.execute(
        f"""SELECT f.id, f.file_path, f.thumbnail_url FROM files f
            JOIN vec_text vt ON vt.file_id=f.id
            WHERE f.id IN ({ph}) AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url!=''""",
        gl).fetchall()[: args.pool]
    pool = {int(r[0]): (r[1], r[2]) for r in rows}
    if len(pool) < args.pool:
        extra = cur.execute(
            f"""SELECT f.id, f.file_path, f.thumbnail_url FROM files f
                JOIN vec_text vt ON vt.file_id=f.id
                WHERE f.id NOT IN ({ph}) AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url!=''
                LIMIT ?""", gl + [args.pool - len(pool)]).fetchall()
        for fid, fp, tu in extra:
            pool[int(fid)] = (fp, tu)
    ids = sorted(pool)
    idx = {f: n for n, f in enumerate(ids)}
    path_of = {f: pool[f][0] for f in ids}
    id_of_path = {v: k for k, v in path_of.items()}
    thumbs = {f: pool[f][1] for f in ids}
    pool_set = set(ids)
    print(f"질의 {len(queries)} / 풀 {len(ids)} (정답 {len(pool_set & gt_all)})")

    # MV
    ph2 = ",".join("?" * len(ids))
    r2 = cur.execute(f"SELECT file_id, embedding FROM vec_text WHERE file_id IN ({ph2})", ids).fetchall()
    dim = len(np.frombuffer(r2[0][1], dtype=np.float32))
    mv = np.zeros((len(ids), dim), dtype=np.float32)
    for fid, blob in r2:
        mv[idx[int(fid)]] = l2(np.frombuffer(blob, dtype=np.float32))

    # VV (SigLIP2) — 현행 visual 축
    print("\n[VV] SigLIP2 인코딩")
    vv = vv_matrix(ids, thumbs, CACHE_DIR / f"siglip2_pool{len(ids)}.npz")

    embedder = get_text_embedding_provider()

    # 비교할 가중치 구성
    schemes = {"uniform": None}          # None → rrf_merge_multi 가 1.0 균등
    for name, preset in WEIGHT_PRESETS.items():
        schemes[f"preset:{name}"] = preset

    Ks = (5, 10, 20)
    acc = {s: {k: {"p": [], "r": []} for k in Ks} for s in schemes}
    n_q = 0

    for q in queries:
        gt = set(q["gt_ids"]) & pool_set
        if not gt:
            continue
        n_q += 1
        qt = q["query"]
        kws = [w for w in (q.get("elements_ko") or []) + (q.get("elements_en") or []) if w]

        qv_vv = l2(search.encoders.vv_encoder.encode_text(qt)) if hasattr(search.encoders, "vv_encoder") else None
        qv_mv = l2(embedder.encode(qt, is_query=True))

        s_vv = vv @ qv_vv if qv_vv is not None else np.zeros(len(ids), dtype=np.float32)
        s_mv = mv @ qv_mv
        o_vv = np.argsort(-s_vv)[:200]
        o_mv = np.argsort(-s_mv)[:200]

        ax = []
        ax.append(("visual", [{"file_path": path_of[ids[i]], "similarity": float(s_vv[i])} for i in o_vv]))
        ax.append(("text_vec", [{"file_path": path_of[ids[i]], "text_similarity": float(s_mv[i])} for i in o_mv]))
        fts = search.fts_search(kws, top_k=200, file_ids=pool_set) if kws else []
        if fts:
            ax.append(("fts", fts))

        for sname, w in schemes.items():
            merged = rrf_merge_multi(ax, weights=w)
            ranked = [id_of_path.get(r["file_path"]) for r in merged]
            ranked = [r for r in ranked if r is not None]
            for k in Ks:
                acc[sname][k]["p"].append(p_at_k(ranked, gt, k))
                acc[sname][k]["r"].append(r_at_k(ranked, gt, k))

    print("\n" + "=" * 70)
    print(f"  질의 {n_q} / 풀 {len(ids)} — 축 구성 동일, 가중치만 변수")
    print("=" * 70)
    summary = {}
    base = {k: float(np.mean(acc["uniform"][k]["p"])) for k in Ks}
    for sname in schemes:
        summary[sname] = {}
        line = []
        for k in Ks:
            p = float(np.mean(acc[sname][k]["p"])); r = float(np.mean(acc[sname][k]["r"]))
            summary[sname][f"P@{k}"] = round(p, 4)
            summary[sname][f"R@{k}"] = round(r, 4)
            line.append(f"P@{k} {p:.3f} ({p-base[k]:+.3f})")
        print(f"  {sname:18s} " + "  ".join(line))

    spread = {k: max(float(np.mean(acc[s][k]["p"])) for s in schemes)
                 - min(float(np.mean(acc[s][k]["p"])) for s in schemes) for k in Ks}
    print(f"\n  프리셋 간 최대 편차: " + "  ".join(f"P@{k} {spread[k]:.3f}" for k in Ks))

    out = OUT_DIR / f"rrf_weights_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "queryset": QUERYSET.name, "pool_size": len(ids), "queries": n_q,
        "summary": summary, "spread": {f"P@{k}": round(v, 4) for k, v in spread.items()},
        "note": ("축 구성(VV=SigLIP2, MV, FTS)과 후보는 모든 구성에서 동일하고 "
                 "rrf_merge_multi 의 weights 만 바꿨다. 편차가 작으면 질의유형별 "
                 "가중치 튜닝이 실질 기여를 못 한다는 뜻이고, WeMM 통합으로 "
                 "이 손잡이를 잃어도 손실이 없다."),
    }, ensure_ascii=False, indent=1))
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
