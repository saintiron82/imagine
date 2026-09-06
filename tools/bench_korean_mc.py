#!/usr/bin/env python3
"""한국어 MC 가설 검증 — 색인이 한국어면 현행 MV 가 회복되는가.

앞선 측정이 진단한 병:
  mc_caption 16,720건 전부 영어. 한글 0건.
  그래서 한국어 질의는 MC 색인에 안 걸리고 폴더 경로에 의존한다.
    한국어 MV P@10 0.287  /  영어 MV P@10 0.457   ← 격차 0.170
    WeMM 은 언어 무관 0.490 / 0.480              ← 격차 0.010

처방이 둘이다:
  A  WeMM 도입      5.3시간 재인코딩 + DB 차원 변경 + RRF 축 손실
  B  VLM 이 한국어도 쓰게 한다   프롬프트 수정

B 가 성립하면 A 의 근거 대부분이 사라진다. 이 스크립트가 B 를 검증한다.

방법: 기존 영어 caption/tags 를 로컬 MLX(Qwen3.5-9B)로 한국어 번역 → 같은
build_document_text 로 문서를 만들어 같은 임베더로 ko-MV 를 생성 → 동일 A/B.
VLM 이 처음부터 한국어로 쓴 것과 완전히 같지는 않지만("번역 ≠ 원생성"),
"색인이 한국어면 한국어 질의가 회복되는가"라는 질문에는 직접 답한다.

3자 비교:
  WeMM     이미지 직접 임베딩 (언어 무관)
  en-MV    현행 (영어 캡션 기반)
  ko-MV    번역된 한국어 캡션 기반   ← 이번에 새로 만드는 것

Usage:
  python tools/bench_korean_mc.py --cache benchmarks/cache/wemm2b_pool700.npz
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

MLX_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

PROMPT = """Translate to Korean. Output ONLY compact JSON, no thinking, no explanation.
Keep proper nouns and codes (nfb03, VS9 etc) as-is.
{{"caption":"...","tags":["...","..."]}}

caption: {caption}
tags: {tags}"""


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def p_at_k(ranked, gt, k):
    return sum(1 for i in ranked[:k] if i in gt) / max(1, k)


def r_at_k(ranked, gt, k):
    return len(set(ranked[:k]) & gt) / max(1, len(gt))


def translate_all(rows, cache_path):
    """영어 caption/tags → 한국어. 캐시가 있으면 재사용."""
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        if set(data) >= {str(r[0]) for r in rows}:
            print(f"  번역 캐시 사용: {cache_path.name}")
            return {int(k): v for k, v in data.items()}

    from mlx_lm import load, generate
    print(f"  MLX 로드: {MLX_MODEL}")
    model, tok = load(MLX_MODEL)

    out = {}
    t0 = time.perf_counter()
    for n, (fid, cap, tags_json) in enumerate(rows):
        try:
            tags = json.loads(tags_json) if tags_json else []
        except Exception:
            tags = []
        prompt = PROMPT.format(caption=cap or "", tags=", ".join(tags))
        msgs = [{"role": "user", "content": prompt}]
        # Qwen3.5 는 기본이 thinking 모드다. 끄지 않으면 생성 토큰을 사고에 다
        # 소진해 JSON 이 안 나온다(실측: 10.33s/건 · 파싱 실패). 끄면 4.04s/건 · 10/10 성공.
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        try:
            resp = generate(model, tok, prompt=text, max_tokens=160, verbose=False)
            s = resp[resp.find("{"): resp.rfind("}") + 1]
            obj = json.loads(s)
            out[fid] = {"caption": obj.get("caption") or "", "tags": obj.get("tags") or []}
        except Exception:
            out[fid] = {"caption": "", "tags": []}
        if (n + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  [{n+1}/{len(rows)}] {el:.0f}s ({el/(n+1):.2f}s/건)", flush=True)
            # 중간 저장 — 중단돼도 여기까지는 재사용된다
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({str(k): v for k, v in out.items()}, ensure_ascii=False))
    el = time.perf_counter() - t0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({str(k): v for k, v in out.items()}, ensure_ascii=False))
    print(f"  번역 완료 {el:.0f}s ({el/len(rows):.2f}s/건) → 캐시 저장")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_DIR / "wemm2b_pool700.npz"))
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    ap.add_argument("--wemm", default="tencent/WeMM-Embedding-2B")
    ap.add_argument("--sample", type=int, default=0, help="번역 대상 표본 수(0=전체)")
    args = ap.parse_args()

    from backend.db.sqlite_client import SQLiteDB
    from backend.vector.text_embedding import get_text_embedding_provider, build_document_text
    from wemm_encoder import WeMMEncoder

    z = np.load(args.cache)
    ids = [int(i) for i in z["ids"]]
    wm = z["mat"]
    if args.sample and args.sample < len(ids):
        # 풀 전체를 줄인다. 번역 대상만 줄이면 미번역분이 영어로 남아
        # ko-MV 에만 유리한 비대칭이 생긴다. 정답을 우선 포함시킨다.
        gtset = {i for q in json.load(open(QUERYSET))["queries"] for i in q["gt_ids"]}
        keep = [i for i in ids if i in gtset][: args.sample]
        keep += [i for i in ids if i not in gtset][: max(0, args.sample - len(keep))]
        keep = sorted(set(keep))
        sel = [ids.index(i) for i in keep]
        wm = wm[sel]
        ids = keep
        print(f"  표본 축소: 풀 {len(ids)} (정답 {len(set(ids)&gtset)})")
    idx = {f: n for n, f in enumerate(ids)}
    pool_set = set(ids)
    print(f"풀 {len(ids)} (WeMM 캐시 재사용)")

    db = SQLiteDB(args.db)
    cur = db.conn.cursor()
    ph = ",".join("?" * len(ids))

    # 기존 en-MV
    rows = cur.execute(f"SELECT file_id, embedding FROM vec_text WHERE file_id IN ({ph})", ids).fetchall()
    dim = len(np.frombuffer(rows[0][1], dtype=np.float32))
    en_mv = np.zeros((len(ids), dim), dtype=np.float32)
    for fid, blob in rows:
        en_mv[idx[int(fid)]] = l2(np.frombuffer(blob, dtype=np.float32))

    # 번역
    print("\n[번역] 영어 caption/tags → 한국어")
    src = cur.execute(
        f"SELECT id, mc_caption, ai_tags FROM files WHERE id IN ({ph})", ids).fetchall()
    ko = translate_all(src, CACHE_DIR / f"ko_mc_pool{len(ids)}.json")

    # ko-MV 생성
    print("\n[ko-MV] 한국어 문서 임베딩")
    embedder = get_text_embedding_provider()
    ko_mv = np.zeros((len(ids), dim), dtype=np.float32)
    t0 = time.perf_counter()
    n_ok = 0
    for fid in ids:
        k = ko.get(fid) or {}
        cap, tags = k.get("caption") or "", k.get("tags") or []
        if not cap and not tags:
            ko_mv[idx[fid]] = en_mv[idx[fid]]   # 번역 실패 → 기존 값 유지
            continue
        doc = build_document_text(caption=cap, tags=tags)
        ko_mv[idx[fid]] = l2(embedder.encode(doc, is_query=False))
        n_ok += 1
    print(f"  {n_ok}/{len(ids)}건 재구성 ({time.perf_counter()-t0:.0f}s)")

    # A/B/C
    enc = WeMMEncoder(args.wemm)
    queries = json.load(open(QUERYSET))["queries"]
    Ks = (5, 10, 20)
    tags_ = ("WeMM", "en_MV", "ko_MV")
    acc = {t: {k: {"p": [], "r": []} for k in Ks} for t in tags_}
    per_query = []

    for q in queries:
        gt = set(q["gt_ids"]) & pool_set
        if not gt:
            continue
        qt = q["query"]
        mats = {"WeMM": (wm, l2(enc.encode_text(qt))),
                "en_MV": (en_mv, l2(embedder.encode(qt, is_query=True))),
                "ko_MV": (ko_mv, l2(embedder.encode(qt, is_query=True)))}
        row = {"query": qt, "gt": len(gt)}
        for t, (mat, qv) in mats.items():
            r = [ids[i] for i in np.argsort(-(mat @ qv))]
            for k in Ks:
                p, rc = p_at_k(r, gt, k), r_at_k(r, gt, k)
                acc[t][k]["p"].append(p); acc[t][k]["r"].append(rc)
                row[f"{t}_P@{k}"] = round(p, 3)
        per_query.append(row)

    print("\n" + "=" * 66)
    print(f"  한국어 질의 {len(per_query)}개 / 풀 {len(ids)}")
    print("=" * 66)
    summary = {}
    for k in Ks:
        summary[f"K={k}"] = {}
        print(f"  K={k}")
        for t in tags_:
            p = float(np.mean(acc[t][k]["p"])); r = float(np.mean(acc[t][k]["r"]))
            summary[f"K={k}"][t] = {"P": round(p, 4), "R": round(r, 4)}
            print(f"    {t:8s} P {p:.3f}  R {r:.3f}")
        gain = summary[f"K={k}"]["ko_MV"]["P"] - summary[f"K={k}"]["en_MV"]["P"]
        left = summary[f"K={k}"]["WeMM"]["P"] - summary[f"K={k}"]["ko_MV"]["P"]
        print(f"    → 한국어화 회복 {gain:+.3f} / 그러고도 WeMM 이 앞선 폭 {left:+.3f}")

    out = OUT_DIR / f"korean_mc_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "queryset": QUERYSET.name, "pool_size": len(ids),
        "translated_ok": n_ok, "summary": summary, "per_query": per_query,
        "note": ("번역(MLX Qwen3.5-9B)으로 만든 ko-MV 는 VLM 이 처음부터 한국어로 "
                 "생성한 것과 동일하지 않다. '색인이 한국어면 회복되는가'의 대리 검증이다. "
                 "풀 제한 실험이라 상대 비교만 유효."),
    }, ensure_ascii=False, indent=1))
    print(f"\n  저장: {out}")


if __name__ == "__main__":
    main()
