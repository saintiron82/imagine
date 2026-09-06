#!/usr/bin/env python3
"""언어 대칭성 — 한국어/영어 질의에서 WeMM 직접 vs 현행 MV.

앞선 측정에서 FTS 컬럼 분포가 언어에 따라 완전히 갈렸다:
  한국어 질의  MC 유래 60% / 파싱(폴더 경로) 44%
  영어 질의    MC 유래 97% / 파싱 2%
즉 현행 시스템은 한국어는 폴더 구조에, 영어는 MC 태그에 의존하는 비대칭이다.

그리고 bench_wemm_vs_mv.py 의 A/B 는 **한국어 질의셋으로만** 돌렸다.
영어에서는 MC 기반 색인이 훨씬 두텁기 때문에 결과가 다를 수 있다 — 그게
WeMM 도입 판단의 마지막 구멍이다.

이 스크립트는 같은 질의 레코드·같은 정답(gt_ids)에 대해 질의 표면만 한국어/영어로
바꿔 A(WeMM 직접) 대 B(현행 MV)를 비교한다. 언어가 유일한 변수다.

WeMM 이미지 임베딩은 benchmarks/cache 의 캐시를 재사용한다(재인코딩 없음).

Usage:
  python tools/bench_lang_symmetry.py --cache benchmarks/cache/wemm2b_pool700.npz
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


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def p_at_k(ranked, gt, k):
    top = ranked[:k]
    return sum(1 for i in top if i in gt) / max(1, len(top))


def r_at_k(ranked, gt, k):
    return len(set(ranked[:k]) & gt) / max(1, len(gt))


def en_query(q):
    """같은 질의를 영어 표면으로. 정답(gt_ids)은 그대로 쓴다."""
    els = q.get("elements_en") or []
    folder = q.get("folder") or ""
    body = " and ".join(els) if els else "images"
    return f"images with {body} in {folder}".strip() if folder else f"images with {body}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "benchmarks/cache/wemm2b_pool700.npz"))
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    ap.add_argument("--wemm", default="tencent/WeMM-Embedding-2B")
    args = ap.parse_args()

    from backend.db.sqlite_client import SQLiteDB
    from backend.vector.text_embedding import get_text_embedding_provider
    from wemm_encoder import WeMMEncoder

    z = np.load(args.cache)
    ids = [int(i) for i in z["ids"]]
    wm = z["mat"]
    idx = {f: n for n, f in enumerate(ids)}
    pool_set = set(ids)
    print(f"캐시 재사용: {Path(args.cache).name}  풀 {len(ids)}  차원 {wm.shape[1]}")

    db = SQLiteDB(args.db)
    cur = db.conn.cursor()
    ph = ",".join("?" * len(ids))
    rows = cur.execute(f"SELECT file_id, embedding FROM vec_text WHERE file_id IN ({ph})", ids).fetchall()
    dim_b = len(np.frombuffer(rows[0][1], dtype=np.float32))
    mv = np.zeros((len(ids), dim_b), dtype=np.float32)
    for fid, blob in rows:
        mv[idx[int(fid)]] = l2(np.frombuffer(blob, dtype=np.float32))
    print(f"MV 행렬 {mv.shape}")

    enc = WeMMEncoder(args.wemm)
    embedder = get_text_embedding_provider()
    queries = json.load(open(QUERYSET))["queries"]

    Ks = (5, 10, 20)
    langs = ("ko", "en")
    acc = {lg: {t: {k: {"p": [], "r": []} for k in Ks} for t in ("A_wemm", "B_mv")} for lg in langs}
    per_query = []

    for q in queries:
        gt = set(q["gt_ids"]) & pool_set
        if not gt:
            continue
        texts = {"ko": q["query"], "en": en_query(q)}
        row = {"ko": q["query"], "en": texts["en"], "gt": len(gt)}
        for lg in langs:
            t = texts[lg]
            sa = wm @ l2(enc.encode_text(t))
            sb = mv @ l2(embedder.encode(t, is_query=True))
            ra = [ids[i] for i in np.argsort(-sa)]
            rb = [ids[i] for i in np.argsort(-sb)]
            for k in Ks:
                for tag, r in (("A_wemm", ra), ("B_mv", rb)):
                    p, rc = p_at_k(r, gt, k), r_at_k(r, gt, k)
                    acc[lg][tag][k]["p"].append(p)
                    acc[lg][tag][k]["r"].append(rc)
                    row[f"{lg}_{tag}_P@{k}"] = round(p, 3)
        per_query.append(row)

    print("\n" + "=" * 66)
    print(f"  결과 — 질의 {len(per_query)} / 풀 {len(ids)}")
    print("=" * 66)
    summary = {}
    for lg in langs:
        label = "한국어" if lg == "ko" else "영어"
        print(f"\n  [{label}]")
        summary[lg] = {}
        for k in Ks:
            a = float(np.mean(acc[lg]["A_wemm"][k]["p"]))
            b = float(np.mean(acc[lg]["B_mv"][k]["p"]))
            ar = float(np.mean(acc[lg]["A_wemm"][k]["r"]))
            br = float(np.mean(acc[lg]["B_mv"][k]["r"]))
            summary[lg][f"P@{k}"] = {"A_wemm": round(a, 4), "B_mv": round(b, 4), "diff": round(a - b, 4)}
            summary[lg][f"R@{k}"] = {"A_wemm": round(ar, 4), "B_mv": round(br, 4), "diff": round(ar - br, 4)}
            print(f"    P@{k:<3} A(WeMM) {a:.3f}  B(MV) {b:.3f}  차 {a-b:+.3f}   "
                  f"R@{k} A {ar:.3f} B {br:.3f} 차 {ar-br:+.3f}")

    # 언어 간 대칭성: 같은 경로가 언어를 바꿔도 같은 성능을 내는가
    print("\n  [언어 대칭성 — P@10 기준]")
    for tag, name in (("A_wemm", "WeMM 직접"), ("B_mv", "현행 MV")):
        ko = float(np.mean(acc["ko"][tag][10]["p"]))
        en = float(np.mean(acc["en"][tag][10]["p"]))
        gap = abs(ko - en)
        print(f"    {name:10s} 한국어 {ko:.3f}  영어 {en:.3f}  격차 {gap:.3f}")

    out = OUT_DIR / f"lang_symmetry_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "queryset": QUERYSET.name, "pool_size": len(ids),
        "wemm_model": args.wemm, "cache": Path(args.cache).name,
        "summary": summary, "per_query": per_query,
        "note": ("영어 질의는 elements_en + folder 로 합성했다(정답은 원본 gt_ids 유지). "
                 "풀 제한 실험이라 절대값이 아니라 A/B·언어 간 상대 비교만 유효."),
    }, ensure_ascii=False, indent=1))
    print(f"\n  저장: {out}")


if __name__ == "__main__":
    main()
