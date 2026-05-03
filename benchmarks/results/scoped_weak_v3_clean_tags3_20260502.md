# Precision@K Benchmark — Real Search Quality

**Date**: 2026-05-02 00:53
**Queries**: 30 (natural Korean)
**DB Size**: 17725 images
**Avg ground truth set**: 25 images (median: 18)

## What This Measures

Not "did we find that exact file?" but **"are the search results relevant?"**

P@5 = 0.60 means: of the top 5 results, 3 are relevant to the query.
This is how users experience search quality.

## Results

| Axis | P@3 | P@5 | P@10 | Role |
|---|:-:|:-:|:-:|---|
| VV | 0.133 | 0.113 (F) | 0.107 | Visual similarity (SigLIP2) |
| MV | 0.044 | 0.047 (F) | 0.043 | Semantic meaning (Qwen3-Embed) |
| FTS | 0.011 | 0.007 (F) | 0.007 | Keyword match (FTS5 BM25) |
| **TRIAXIS** | **0.633** | **0.547 (D)** | **0.463** | **VV + MV + FTS → RRF** |

**Fusion Lift P@5**: 0.113 → 0.547 (**+382.5%**)

## Interpretation

Triaxis P@5 = 0.547 means:
- Of top 5 results, **2.7 images are relevant** on average
- User sees 55% relevant results in the first page

**Production Level**: Production (주력 검색 엔진 가능)

---

## Per-Query Detail (top 20)

| Query | GT Size | P@5 | Hits@5 | VV P@5 | MV P@5 | FTS P@5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| #08에서 밤과 달 있는 이미지 | 5 | **1.00** | 5 | 0.00 | 0.00 | 0.00 |
| #07에서 캐릭터과 방 있는 이미지 | 18 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| #09에서 하늘과 캐릭터 있는 이미지 | 99 | **1.00** | 5 | 0.20 | 0.00 | 0.00 |
| 로네느의집에서 그림과 방 있는 이미지 | 19 | **0.40** | 2 | 0.00 | 0.00 | 0.00 |
| 홍콩사무실에서 소파과 창문 있는 이미지 | 2 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| bg에서 train과 하늘 있는 이미지 | 2 | **0.00** | 0 | 0.00 | 0.00 | 0.00 |
| #3에서 하늘과 밤 있는 이미지 | 42 | **0.00** | 0 | 0.00 | 0.40 | 0.00 |
| #01에서 교실과 캐릭터 있는 이미지 | 20 | **0.80** | 4 | 0.00 | 0.00 | 0.00 |
| #01에서 밤과 캐릭터 있는 이미지 | 18 | **0.40** | 2 | 0.00 | 0.00 | 0.00 |
| #3에서 창문과 주방 있는 이미지 | 11 | **0.00** | 0 | 0.00 | 0.00 | 0.00 |
| 크랑베르무에서 숲과 밤 있는 이미지 | 46 | **1.00** | 5 | 0.00 | 0.00 | 0.00 |
| #2에서 발코니과 창문 있는 이미지 | 3 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 크랑베르무에서 하늘과 pillar 있는 이미지 | 27 | **0.80** | 4 | 0.20 | 0.00 | 0.00 |
| 작품 쫑에서 검과 armor 있는 이미지 | 32 | **1.00** | 5 | 0.20 | 0.00 | 0.00 |
| 안나의집에서 커튼과 창문 있는 이미지 | 8 | **0.20** | 1 | 0.20 | 0.00 | 0.00 |
| 작품 쫑에서 밤과 벽 있는 이미지 | 18 | **0.40** | 2 | 0.00 | 0.00 | 0.00 |
| 늪지대에서 숲과 fog 있는 이미지 | 30 | **1.00** | 5 | 0.60 | 0.00 | 0.00 |
| #08에서 하늘과 캐릭터 있는 이미지 | 57 | **1.00** | 5 | 0.00 | 0.00 | 0.00 |
| #05에서 하늘과 캐릭터 있는 이미지 | 24 | **1.00** | 5 | 0.60 | 0.00 | 0.00 |
| #02에서 밤과 숲 있는 이미지 | 41 | **1.00** | 5 | 0.00 | 0.00 | 0.00 |

---

## Query Examples

- `#08에서 밤과 달 있는 이미지` — GT: 5 images, elements: ['night', 'moon']
- `#07에서 캐릭터과 방 있는 이미지` — GT: 18 images, elements: ['character', 'room']
- `#09에서 하늘과 캐릭터 있는 이미지` — GT: 99 images, elements: ['sky', 'character']
- `로네느의집에서 그림과 방 있는 이미지` — GT: 19 images, elements: ['painting', 'room']
- `홍콩사무실에서 소파과 창문 있는 이미지` — GT: 2 images, elements: ['sofa', 'window']

---

## Timing

- init: 0.3s
- query_gen: 0.9s
- vv: 57.0s
- mv: 47.8s
- fts: 0.0s
- triaxis: 166.4s
