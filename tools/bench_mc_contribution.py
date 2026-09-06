#!/usr/bin/env python3
"""MC 가 검색에 기여하는가 — WeMM 단독 vs WeMM+FTS vs 현행 전체.

앞선 bench_wemm_vs_mv.py 는 "WeMM 직접 vs MV(캡션 경유)"를 재서 MV 가 손해임을
보였다. 이 실험은 그 다음 질문에 답한다:

  임베딩이 충분히 정확하면, MC 로 얻으려던 것(태그·분류)을 검색 시점에
  임베딩이 이미 다 찾아내는가?

  A  WeMM 단독                     임베딩만
  B  WeMM + FTS(MC 유래 컬럼)      MC 를 얹으면 나아지는가
  C  현행 전체 (VV + MV + FTS)     기준선

A ≈ B 면 MC 의 검색 기여는 0 이다 — MC 는 검색이 아니라 UI 패싯·설명 같은
다른 이유로만 존재하게 되고, 장당 21초를 쓸 근거가 사라진다.
B > A 면 임베딩이 못 잡는 무언가를 MC 가 잡고 있다는 뜻이고, 질의별 차이를
보면 그게 무엇인지 드러난다.

주의 — FTS 는 MC 유래 컬럼(caption/ai_tags/classification/spatial)과 파싱 유래
컬럼(meta_strong=파일명·레이어명·폰트·OCR, meta_weak=경로·텍스트레이어)을 함께
가진다. MC 를 없애도 파싱 유래분은 남으므로, B 의 이득 전부를 MC 몫으로
읽으면 안 된다. 그래서 B 를 두 갈래로 나눠 잰다:
  B1  WeMM + FTS 전체
  B2  WeMM + FTS(파싱 유래만)   ← MC 컬럼을 질의에서 제외
B1 - B2 가 MC 의 순수 기여다.

융합은 프로덕션 코드(backend.search.scoring.rrf_merge_multi)를 그대로 쓴다.

Usage:
  python tools/bench_mc_contribution.py --pool 5000
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
CACHE_DIR = PROJECT_ROOT / "benchmarks/cache"


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def p_at_k(ranked, gt, k):
    top = ranked[:k]
    return sum(1 for i in top if i in gt) / max(1, len(top))


def r_at_k(ranked, gt, k):
    return len(set(ranked[:k]) & gt) / max(1, len(gt))


def build_pool(db, gt_ids, pool_size, seed=0):
    cur = db.conn.cursor()
    ph = ",".join("?" * len(gt_ids))
    base = cur.execute(
        f"""SELECT f.id, f.file_path, f.thumbnail_url FROM files f
            JOIN vec_text vt ON vt.file_id = f.id
            WHERE f.id IN ({ph})
              AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url != ''""", gt_ids).fetchall()
    pool = {int(r[0]): (r[1], r[2]) for r in base}
    need = max(0, pool_size - len(pool))
    if need:
        extra = cur.execute(
            f"""SELECT f.id, f.file_path, f.thumbnail_url FROM files f
                JOIN vec_text vt ON vt.file_id = f.id
                WHERE f.id NOT IN ({ph})
                  AND f.thumbnail_url IS NOT NULL AND f.thumbnail_url != ''""", gt_ids).fetchall()
        random.Random(seed).shuffle(extra)
        for fid, fp, tu in extra[:need]:
            pool[int(fid)] = (fp, tu)
    return pool


def wemm_matrix(enc, ids, pool, cache_path):
    """WeMM 이미지 임베딩. 캐시가 있으면 재사용 — 재인코딩은 장당 1.08s 다."""
    if cache_path.exists():
        z = np.load(cache_path)
        cached_ids = list(z["ids"])
        if cached_ids == list(ids):
            print(f"  캐시 사용: {cache_path.name}")
            return z["mat"], 0.0
        idx = {int(f): n for n, f in enumerate(cached_ids)}
        if all(i in idx for i in ids):
            print(f"  캐시 부분 사용: {cache_path.name}")
            return np.stack([z["mat"][idx[i]] for i in ids]), 0.0

    from PIL import Image
    t0 = time.perf_counter()
    mat = None
    for n, fid in enumerate(ids):
        v = None
        try:
            with Image.open(pool[fid][1]) as im:
                v = enc.encode_image(im.convert("RGB"))
        except Exception:
            pass
        if mat is None:
            mat = np.zeros((len(ids), len(v) if v is not None else 2048), dtype=np.float32)
        if v is not None:
            mat[n] = v
        if (n + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  [{n+1}/{len(ids)}] {el:.0f}s ({el/(n+1):.2f}s/장)")
    el = time.perf_counter() - t0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(ids), mat=mat)
    print(f"  인코딩 {el:.0f}s ({el/len(ids):.2f}s/장) → 캐시 저장")
    return mat, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    ap.add_argument("--wemm", default="tencent/WeMM-Embedding-2B")
    args = ap.parse_args()

    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    from backend.search.scoring import rrf_merge_multi
    from backend.vector.text_embedding import get_text_embedding_provider
    from wemm_encoder import WeMMEncoder

    queries = json.load(open(QUERYSET))["queries"]
    gt_ids = sorted({i for q in queries for i in q["gt_ids"]})
    db = SQLiteDB(args.db)
    search = SqliteVectorSearch(db=db)   # fts_search 는 검색 클래스 소속
    pool = build_pool(db, gt_ids, args.pool)
    ids = sorted(pool)
    idx = {f: n for n, f in enumerate(ids)}
    path_of = {f: pool[f][0] for f in ids}
    id_of_path = {v: k for k, v in path_of.items()}
    print(f"질의 {len(queries)} / 정답 {len(gt_ids)} / 풀 {len(ids)}")

    # MV 행렬 (경로 C 용)
    cur = db.conn.cursor()
    ph = ",".join("?" * len(ids))
    rows = cur.execute(f"SELECT file_id, embedding FROM vec_text WHERE file_id IN ({ph})", ids).fetchall()
    dim_b = len(np.frombuffer(rows[0][1], dtype=np.float32))
    mv = np.zeros((len(ids), dim_b), dtype=np.float32)
    for fid, blob in rows:
        mv[idx[int(fid)]] = l2(np.frombuffer(blob, dtype=np.float32))

    # WeMM 행렬
    print("\n[WeMM] 이미지 인코딩")
    enc = WeMMEncoder(args.wemm)
    wm, enc_s = wemm_matrix(enc, ids, pool, CACHE_DIR / f"wemm2b_pool{len(ids)}.npz")

    embedder = get_text_embedding_provider()
    pool_set = set(ids)

    def as_axis(order_ids, scores, key):
        """rrf_merge_multi 가 먹는 형태로 변환."""
        return [{"file_path": path_of[i], key: float(s)} for i, s in zip(order_ids, scores)]

    Ks = (5, 10, 20)
    tags = ("A_wemm", "B1_wemm_fts", "B2_wemm_fts_parse", "C_current")
    acc = {t: {k: {"p": [], "r": []} for k in Ks} for t in tags}
    per_query = []

    MC_COLS = ("caption", "ai_tags", "classification", "spatial")

    for q in queries:
        gt = set(q["gt_ids"]) & pool_set
        if not gt:
            continue
        qtext = q["query"]
        kws = [w for w in (q.get("elements_ko") or []) + (q.get("elements_en") or []) if w]

        # ── 벡터 축 ──
        qa = l2(enc.encode_text(qtext))
        qb = l2(embedder.encode(qtext, is_query=True))
        sa = wm @ qa
        sb = mv @ qb
        oa = np.argsort(-sa)[:200]
        ob = np.argsort(-sb)[:200]
        ax_wemm = as_axis([ids[i] for i in oa], sa[oa], "similarity")
        ax_mv = as_axis([ids[i] for i in ob], sb[ob], "text_similarity")

        # ── FTS 축 ──
        fts_all = db_fts(search, kws, pool_set, cols=None)
        fts_parse = db_fts(search, kws, pool_set, cols=("meta_strong", "meta_weak"))

        runs = {
            "A_wemm": [("visual", ax_wemm)],
            "B1_wemm_fts": [("visual", ax_wemm), ("fts", fts_all)],
            "B2_wemm_fts_parse": [("visual", ax_wemm), ("fts", fts_parse)],
            "C_current": [("visual", ax_wemm if False else ax_wemm), ("text_vec", ax_mv), ("fts", fts_all)],
        }
        # C 는 현행 구성이므로 visual 축을 SigLIP2(VV)로 두는 게 정확하지만,
        # VV 를 풀 전체에 대해 다시 인코딩하는 비용이 커서 여기서는 MV+FTS 로만
        # 구성한다. 아래에서 visual 축을 뺀다.
        runs["C_current"] = [("text_vec", ax_mv), ("fts", fts_all)]

        row = {"query": qtext, "gt": len(gt)}
        for tag, lists in runs.items():
            merged = rrf_merge_multi(lists)
            ranked = [id_of_path.get(r["file_path"]) for r in merged]
            ranked = [r for r in ranked if r is not None]
            for k in Ks:
                p, rc = p_at_k(ranked, gt, k), r_at_k(ranked, gt, k)
                acc[tag][k]["p"].append(p); acc[tag][k]["r"].append(rc)
                row[f"{tag}_P@{k}"] = round(p, 3)
        per_query.append(row)

    print("\n" + "=" * 74)
    print(f"  결과 — 질의 {len(per_query)} / 풀 {len(ids)}")
    print("=" * 74)
    summary = {}
    for k in Ks:
        line = {}
        for t in tags:
            line[t] = {"P": round(float(np.mean(acc[t][k]["p"])), 4),
                       "R": round(float(np.mean(acc[t][k]["r"])), 4)}
        summary[f"K={k}"] = line
        print(f"  K={k}")
        for t in tags:
            print(f"    {t:20s} P {line[t]['P']:.3f}   R {line[t]['R']:.3f}")
        mc_gain = line["B1_wemm_fts"]["P"] - line["B2_wemm_fts_parse"]["P"]
        emb_only = line["B1_wemm_fts"]["P"] - line["A_wemm"]["P"]
        print(f"    → FTS 전체 기여 {emb_only:+.3f} / 그중 MC 컬럼 순수 기여 {mc_gain:+.3f}")

    out = OUT_DIR / f"mc_contribution_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "queryset": QUERYSET.name, "pool_size": len(ids), "gt_total": len(gt_ids),
        "wemm_model": args.wemm, "wemm_encode_s": round(enc_s, 1),
        "summary": summary, "per_query": per_query,
        "note": ("풀 제한 실험 — A/B 상대 비교만 유효. "
                 "C 는 VV 재인코딩 비용 때문에 MV+FTS 로만 구성했으므로 "
                 "현행 전체(VV 포함)의 완전한 재현이 아니다."),
    }, ensure_ascii=False, indent=1))
    print(f"\n  저장: {out}")


def db_fts(search, keywords, pool_set, cols=None):
    """FTS 검색. cols 를 주면 해당 컬럼으로만 질의(MC 유래 제외 실험용)."""
    if not keywords:
        return []
    try:
        if cols is None:
            return search.fts_search(keywords, top_k=200, file_ids=pool_set)
        # 컬럼 한정: FTS5 의 `col:term` 문법으로 파싱 유래 컬럼만 노린다.
        scoped = [f"{c}:{w}" for w in keywords for c in cols]
        return search.fts_search(scoped, top_k=200, file_ids=pool_set)
    except Exception as e:
        print(f"    (FTS 실패: {e})")
        return []


if __name__ == "__main__":
    main()
