# Precision@K Benchmark — Real Search Quality

**Date**: 2026-05-02 00:07
**Queries**: 30 (natural Korean)
**DB Size**: 17725 images
**Avg ground truth set**: 11 images (median: 5)

## What This Measures

Not "did we find that exact file?" but **"are the search results relevant?"**

P@5 = 0.60 means: of the top 5 results, 3 are relevant to the query.
This is how users experience search quality.

## Results

| Axis | P@3 | P@5 | P@10 | Role |
|---|:-:|:-:|:-:|---|
| VV | 0.078 | 0.087 (F) | 0.077 | Visual similarity (SigLIP2) |
| MV | 0.056 | 0.073 (F) | 0.063 | Semantic meaning (Qwen3-Embed) |
| FTS | 0.011 | 0.013 (F) | 0.013 | Keyword match (FTS5 BM25) |
| **TRIAXIS** | **0.500** | **0.420 (D)** | **0.310** | **VV + MV + FTS → RRF** |

**Fusion Lift P@5**: 0.087 → 0.420 (**+384.4%**)

## Interpretation

Triaxis P@5 = 0.420 means:
- Of top 5 results, **2.1 images are relevant** on average
- User sees 42% relevant results in the first page

**Production Level**: 초기 Production (탐색/브라우징)

---

## Per-Query Detail (top 20)

| Query | GT Size | P@5 | Hits@5 | VV P@5 | MV P@5 | FTS P@5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 로네느의집에서 golden_hour과 rural 있는 이미지 | 3 | **0.60** | 3 | 0.40 | 0.20 | 0.00 |
| 안나의집에서 library과 books 있는 이미지 | 2 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 라투루의저택.장미.정원에서 정원과 Rain 있는 이미지 | 65 | **1.00** | 5 | 0.80 | 0.00 | 0.00 |
| 기절용사와 암살공주에서 하늘과 일몰 있는 이미지 | 31 | **0.60** | 3 | 0.00 | 0.00 | 0.00 |
| #3에서 line art과 hair 있는 이미지 | 10 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 사카모토에서 강과 숲 있는 이미지 | 23 | **1.00** | 5 | 0.20 | 0.00 | 0.00 |
| 크랑베르무에서 하늘과 grayscale 있는 이미지 | 4 | **0.80** | 4 | 0.00 | 0.00 | 0.00 |
| 로네느의집 실험실에서 occult과 thumb_33767.png 있는 이 | 9 | **0.00** | 0 | 0.00 | 0.60 | 0.20 |
| #1에서 주방과 창문 있는 이미지 | 2 | **0.20** | 1 | 0.00 | 0.00 | 0.00 |
| 유적지에서 geometric과 pattern 있는 이미지 | 4 | **0.00** | 0 | 0.00 | 0.00 | 0.00 |
| 찾집.술집에서 창문과 의자 있는 이미지 | 7 | **0.00** | 0 | 0.00 | 0.20 | 0.00 |
| 크랑베르무에서 dragon과 sci-fi 있는 이미지 | 4 | **0.40** | 2 | 0.00 | 0.20 | 0.00 |
| #01에서 캐릭터과 복도 있는 이미지 | 5 | **0.00** | 0 | 0.00 | 0.00 | 0.00 |
| #08에서 문과 벽 있는 이미지 | 22 | **0.40** | 2 | 0.00 | 0.00 | 0.00 |
| #1에서 metal과 texture 있는 이미지 | 3 | **0.20** | 1 | 0.20 | 0.00 | 0.00 |
| 작품 쫑에서 밤과 fog 있는 이미지 | 5 | **0.60** | 3 | 0.00 | 0.00 | 0.00 |
| #03에서 crowd과 stadium 있는 이미지 | 6 | **1.00** | 5 | 0.20 | 0.40 | 0.00 |
| 실내소품에서 shelves과 drinks 있는 이미지 | 3 | **0.00** | 0 | 0.00 | 0.00 | 0.00 |
| 만게츠의집에서 girl과 책장 있는 이미지 | 8 | **0.80** | 4 | 0.00 | 0.00 | 0.00 |
| #01에서 peaceful과 nature 있는 이미지 | 2 | **0.40** | 2 | 0.00 | 0.00 | 0.00 |

---

## Query Examples

- `로네느의집에서 golden_hour과 rural 있는 이미지` — GT: 3 images, elements: ['golden_hour', 'rural']
- `안나의집에서 library과 books 있는 이미지` — GT: 2 images, elements: ['library', 'books']
- `라투루의저택.장미.정원에서 정원과 Rain 있는 이미지` — GT: 65 images, elements: ['Garden', 'Rain']
- `기절용사와 암살공주에서 하늘과 일몰 있는 이미지` — GT: 31 images, elements: ['sky', 'sunset']
- `#3에서 line art과 hair 있는 이미지` — GT: 10 images, elements: ['line art', 'hair']

---

## Timing

- init: 0.3s
- query_gen: 0.7s
- vv: 58.2s
- mv: 54.1s
- fts: 0.1s
- triaxis: 226.3s
