# Precision@K Benchmark — Real Search Quality

**Date**: 2026-05-01 22:57
**Queries**: 30 (natural Korean)
**DB Size**: 17725 images
**Avg ground truth set**: 16 images (median: 9)

## What This Measures

Not "did we find that exact file?" but **"are the search results relevant?"**

P@5 = 0.60 means: of the top 5 results, 3 are relevant to the query.
This is how users experience search quality.

## Results

| Axis | P@3 | P@5 | P@10 | Role |
|---|:-:|:-:|:-:|---|
| VV | 0.122 | 0.107 (F) | 0.087 | Visual similarity (SigLIP2) |
| MV | 0.189 | 0.160 (F) | 0.120 | Semantic meaning (Qwen3-Embed) |
| FTS | 0.122 | 0.087 (F) | 0.073 | Keyword match (FTS5 BM25) |
| **TRIAXIS** | **0.556** | **0.420 (D)** | **0.263** | **VV + MV + FTS → RRF** |

**Fusion Lift P@5**: 0.160 → 0.420 (**+162.5%**)

## Interpretation

Triaxis P@5 = 0.420 means:
- Of top 5 results, **2.1 images are relevant** on average
- User sees 42% relevant results in the first page

**Production Level**: 초기 Production (탐색/브라우징)

---

## Per-Query Detail (top 20)

| Query | GT Size | P@5 | Hits@5 | VV P@5 | MV P@5 | FTS P@5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 만게츠의집에서 wooden floor과 warm lighting 있는 이 | 10 | **0.20** | 1 | 0.20 | 0.00 | 0.00 |
| #04에서 twilight과 digital art 있는 이미지 | 6 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 호텔실내에서 창문과 vase 있는 이미지 | 23 | **0.80** | 4 | 0.40 | 0.40 | 0.00 |
| 크랑베르무에서 불과 digital art 있는 이미지 | 13 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| #10에서 giant figure과 cracked sphere 있는 이미 | 6 | **0.40** | 2 | 0.00 | 0.20 | 0.20 |
| 크랑베르무에서 textured과 industrial 있는 이미지 | 7 | **0.40** | 2 | 0.00 | 0.20 | 0.00 |
| 로네느의집 실험실에서 Corkboard과 Ancient 있는 이미지 | 2 | **0.00** | 0 | 0.00 | 0.00 | 0.20 |
| 고아원에서 stone_background과 dark 있는 이미지 | 2 | **0.00** | 0 | 0.00 | 0.00 | 0.00 |
| #1에서 shelves과 bins 있는 이미지 | 4 | **0.60** | 3 | 0.00 | 0.00 | 0.60 |
| 기절용사와 암살공주에서 stalagmite과 cave 있는 이미지 | 3 | **0.60** | 3 | 0.00 | 0.40 | 0.00 |
| 신게츠의집에서 선반과 책상 있는 이미지 | 2 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 늪지대에서 절벽과 fog 있는 이미지 | 5 | **0.60** | 3 | 0.20 | 0.20 | 0.00 |
| 잡것들에서 apron과 muscular build 있는 이미지 | 8 | **0.60** | 3 | 0.20 | 0.20 | 0.20 |
| BG.참고에서 하늘과 snow 있는 이미지 | 45 | **0.60** | 3 | 0.60 | 0.60 | 0.20 |
| 유적지에서 brick wall과 애니메이션 있는 이미지 | 16 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 이미지에서 cat ears과 texture 있는 이미지 | 2 | **0.20** | 1 | 0.00 | 0.40 | 0.20 |
| 크랑베르무에서 cosmic과 burst 있는 이미지 | 9 | **1.00** | 5 | 0.00 | 0.40 | 0.00 |
| #05에서 다리과 일몰 있는 이미지 | 31 | **0.80** | 4 | 0.80 | 0.00 | 0.00 |
| #03에서 crowd과 stage 있는 이미지 | 14 | **1.00** | 5 | 0.40 | 0.00 | 0.00 |
| 크랑베르무에서 물과 일몰 있는 이미지 | 26 | **0.60** | 3 | 0.40 | 0.00 | 0.00 |

---

## Query Examples

- `만게츠의집에서 wooden floor과 warm lighting 있는 이미지` — GT: 10 images, elements: ['wooden floor', 'warm lighting']
- `#04에서 twilight과 digital art 있는 이미지` — GT: 6 images, elements: ['twilight', 'digital art']
- `호텔실내에서 창문과 vase 있는 이미지` — GT: 23 images, elements: ['window', 'vase']
- `크랑베르무에서 불과 digital art 있는 이미지` — GT: 13 images, elements: ['fire', 'digital art']
- `#10에서 giant figure과 cracked sphere 있는 이미지` — GT: 6 images, elements: ['giant figure', 'cracked sphere']

---

## Timing

- init: 0.4s
- query_gen: 0.7s
- vv: 73.9s
- mv: 45.2s
- fts: 0.1s
- triaxis: 133.8s
