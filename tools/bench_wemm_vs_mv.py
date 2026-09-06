#!/usr/bin/env python3
"""WeMM 직접 검색 vs 현행 MV(VLM 캡션 경유) A/B.

물음: "MC 를 위해 MV 를 안 해도 되는가" — 즉 이미지를 캡션으로 바꿔 임베딩하는
우회로가 실제로 손해인가.

  경로 A (WeMM 직접) : 질의 →[WeMM 텍스트]→ 벡터 ─┐
                       이미지 →[WeMM 이미지]→ 벡터 ─┴→ 코사인 랭킹
  경로 B (현행 MV)   : 질의 →[Qwen3-Embedding]→ 벡터 ─┐
                       이미지 →[VLM]→ MC →[동일 임베더]→ vec_text ─┴→ 코사인 랭킹

두 경로는 **같은 후보 풀·같은 질의·같은 정답 라벨**로 평가한다. 풀을 제한하는
이유는 WeMM 이미지 인코딩이 느려(≈0.6s/장) 17,726건 전수가 비현실적이기 때문이다.
따라서 절대 P@K 는 전수 기준선(frozen_30_scoped_v1 P@5 0.74)과 비교할 수 없고,
**A 와 B 의 상대 비교만** 유효하다.

vv_quality 벤치와 다른 점: 저기서는 두 인코더가 같은 VLM 캡션을 입력으로 받았다.
여기서는 A 가 VLM 을 아예 거치지 않는다 — 그게 이 실험의 핵심이다.

Usage:
  python tools/bench_wemm_vs_mv.py --pool 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUERYSET = PROJECT_ROOT / "benchmarks/querysets/frozen_30_scoped_v1.json"
OUT_DIR = PROJECT_ROOT / "benchmarks/results"


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def precision_at_k(ranked_ids, gt, k):
    top = ranked_ids[:k]
    return sum(1 for i in top if i in gt) / max(1, len(top))


def recall_at_k(ranked_ids, gt, k):
    top = set(ranked_ids[:k])
    return len(top & gt) / max(1, len(gt))


def build_pool(db, gt_ids, pool_size, seed=0):
    """정답 전부 + 나머지에서 뽑은 방해 후보. 썸네일이 있는 것만."""
    cur = db.conn.cursor()
    ph = ",".join("?" * len(gt_ids))
    rows = cur.execute(
        f"""SELECT f.id, f.thumbnail_url FROM files f
            JOIN vec_text vt ON vt.file_id = f.id
            WHERE f.id IN ({ph})
              AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url != ''""",
        gt_ids,
    ).fetchall()
    pool = {int(r[0]): r[1] for r in rows}

    need = max(0, pool_size - len(pool))
    if need:
        extra = cur.execute(
            f"""SELECT f.id, f.thumbnail_url FROM files f
                JOIN vec_text vt ON vt.file_id = f.id
                WHERE f.id NOT IN ({ph})
                  AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url != ''""",
            gt_ids,
        ).fetchall()
        random.Random(seed).shuffle(extra)
        for fid, url in extra[:need]:
            pool[int(fid)] = url
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2000, help="후보 풀 크기(정답 포함)")
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    ap.add_argument("--wemm", default="tencent/WeMM-Embedding-2B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from backend.db.sqlite_client import SQLiteDB
    from PIL import Image

    queries = json.load(open(QUERYSET))["queries"]
    gt_ids = sorted({i for q in queries for i in q["gt_ids"]})
    print(f"질의 {len(queries)}개 / 고유 정답 {len(gt_ids)}개")

    db = SQLiteDB(args.db)
    pool = build_pool(db, gt_ids, args.pool)
    ids = sorted(pool)
    idx = {fid: n for n, fid in enumerate(ids)}
    print(f"후보 풀 {len(ids)}개 (정답 {len(set(ids) & set(gt_ids))} + 방해 {len(ids)-len(set(ids)&set(gt_ids))})")

    # ── 경로 B: 현행 MV ─────────────────────────────────────
    print("\n[B] 현행 MV — vec_text 적재 + 질의 임베딩")
    t0 = time.perf_counter()
    cur = db.conn.cursor()
    ph = ",".join("?" * len(ids))
    rows = cur.execute(
        f"SELECT file_id, embedding FROM vec_text WHERE file_id IN ({ph})", ids
    ).fetchall()
    mv = np.zeros((len(ids), 0), dtype=np.float32)
    vecs = {}
    for fid, blob in rows:
        vecs[int(fid)] = l2(np.frombuffer(blob, dtype=np.float32))
    dim_b = len(next(iter(vecs.values())))
    mv = np.zeros((len(ids), dim_b), dtype=np.float32)
    for fid, v in vecs.items():
        mv[idx[fid]] = v
    print(f"  MV 행렬 {mv.shape}  ({time.perf_counter()-t0:.1f}s)")

    from backend.vector.text_embedding import get_text_embedding_provider
    embedder = get_text_embedding_provider()
    print(f"  질의 임베더: {embedder}")

    # ── 경로 A: WeMM ────────────────────────────────────────
    print("\n[A] WeMM — 풀 이미지 인코딩")
    from wemm_encoder import WeMMEncoder
    enc = WeMMEncoder(args.wemm)
    t0 = time.perf_counter()
    wm = None
    for n, fid in enumerate(ids):
        try:
            with Image.open(pool[fid]) as im:
                v = enc.encode_image(im.convert("RGB"))
        except Exception as e:
            v = None
        if wm is None:
            wm = np.zeros((len(ids), len(v) if v is not None else 2048), dtype=np.float32)
        if v is not None:
            wm[n] = v
        if (n + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  [{n+1}/{len(ids)}] {el:.0f}s  ({el/(n+1):.2f}s/장)")
    enc_s = time.perf_counter() - t0
    print(f"  완료 {enc_s:.0f}s ({enc_s/len(ids):.2f}s/장)")

    # ── 평가 ────────────────────────────────────────────────
    Ks = (5, 10, 20)
    res = {"A_wemm": {k: {"p": [], "r": []} for k in Ks},
           "B_mv": {k: {"p": [], "r": []} for k in Ks}}
    per_query = []

    for q in queries:
        gt = set(q["gt_ids"]) & set(ids)
        if not gt:
            continue
        qa = l2(enc.encode_text(q["query"]))
        qb = l2(embedder.encode(q["query"], is_query=True))

        sa = wm @ qa
        sb = mv @ qb
        ra = [ids[i] for i in np.argsort(-sa)]
        rb = [ids[i] for i in np.argsort(-sb)]

        row = {"query": q["query"], "gt": len(gt)}
        for k in Ks:
            for tag, r in (("A_wemm", ra), ("B_mv", rb)):
                p = precision_at_k(r, gt, k); rc = recall_at_k(r, gt, k)
                res[tag][k]["p"].append(p); res[tag][k]["r"].append(rc)
                row[f"{tag}_P@{k}"] = round(p, 3)
                row[f"{tag}_R@{k}"] = round(rc, 3)
        per_query.append(row)

    print("\n" + "=" * 62)
    print(f"  결과 — 질의 {len(per_query)}개 / 풀 {len(ids)}개")
    print("=" * 62)
    summary = {}
    for k in Ks:
        a_p = float(np.mean(res["A_wemm"][k]["p"])); b_p = float(np.mean(res["B_mv"][k]["p"]))
        a_r = float(np.mean(res["A_wemm"][k]["r"])); b_r = float(np.mean(res["B_mv"][k]["r"]))
        summary[f"P@{k}"] = {"A_wemm": round(a_p, 4), "B_mv": round(b_p, 4), "diff": round(a_p - b_p, 4)}
        summary[f"R@{k}"] = {"A_wemm": round(a_r, 4), "B_mv": round(b_r, 4), "diff": round(a_r - b_r, 4)}
        print(f"  P@{k:<3} A(WeMM) {a_p:.3f}   B(MV) {b_p:.3f}   차 {a_p-b_p:+.3f}")
        print(f"  R@{k:<3} A(WeMM) {a_r:.3f}   B(MV) {b_r:.3f}   차 {a_r-b_r:+.3f}")

    wins = sum(1 for r in per_query if r["A_wemm_P@5"] > r["B_mv_P@5"])
    losses = sum(1 for r in per_query if r["A_wemm_P@5"] < r["B_mv_P@5"])
    print(f"\n  P@5 기준: A 승 {wins} / B 승 {losses} / 무 {len(per_query)-wins-losses}")

    out = Path(args.out) if args.out else OUT_DIR / f"wemm_vs_mv_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "queryset": QUERYSET.name,
        "pool_size": len(ids), "gt_total": len(gt_ids),
        "wemm_model": args.wemm, "wemm_dim": int(wm.shape[1]), "mv_dim": int(dim_b),
        "wemm_encode_s": round(enc_s, 1), "wemm_s_per_image": round(enc_s / len(ids), 3),
        "summary": summary, "wins_A": wins, "wins_B": losses,
        "per_query": per_query,
        "note": "풀 제한 실험 — 절대 P@K 를 전수 기준선과 비교하지 말 것. A/B 상대 비교만 유효.",
    }, ensure_ascii=False, indent=1))
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
