# Triaxis Full-Pipeline Search Quality Benchmark

**Date**: 2026-04-04 22:15
**Queries**: 50
**DB Size**: 11174 images

## Query Complexity Levels

| Level | Pipeline Stages | Example |
|---|---|---|
| **simple** | VV + MV + FTS → RRF | `peaceful city street_view` |
| **scoped** | + LLM Decomposer + Scope Filter | `background 중에서 peaceful city street_view` |
| **complex** | + Negative Filter + Exclude | `background에서 peaceful city street_view 찾아줘, door 제외` |

---

## SIMPLE

| Axis | R@1 | R@3 | R@5 | R@10 | MRR |
|---|:-:|:-:|:-:|:-:|:-:|
| VV | 0.060 | 0.100 | 0.140 | 0.160 | 0.094 |
| MV | 0.120 | 0.200 | 0.240 | 0.300 | 0.181 |
| FTS | 0.000 | 0.000 | 0.000 | 0.020 | 0.003 |
| **TRIAXIS** | **0.100** | **0.160** | **0.180** | **0.280** | **0.153** |

**Fusion Lift**: R@1 0.120→0.100 (**-16.7%**) | R@5 0.240→0.180 (**-25.0%**)

---

## SCOPED

| Axis | R@1 | R@3 | R@5 | R@10 | MRR |
|---|:-:|:-:|:-:|:-:|:-:|
| VV | 0.040 | 0.100 | 0.100 | 0.140 | 0.074 |
| MV | 0.220 | 0.320 | 0.340 | 0.400 | 0.289 |
| FTS | 0.000 | 0.020 | 0.020 | 0.020 | 0.008 |
| **TRIAXIS** | **0.080** | **0.160** | **0.240** | **0.280** | **0.148** |

**Fusion Lift**: R@1 0.220→0.080 (**-63.6%**) | R@5 0.340→0.240 (**-29.4%**)

---

## COMPLEX

| Axis | R@1 | R@3 | R@5 | R@10 | MRR |
|---|:-:|:-:|:-:|:-:|:-:|
| VV | 0.040 | 0.080 | 0.100 | 0.100 | 0.064 |
| MV | 0.060 | 0.160 | 0.220 | 0.240 | 0.126 |
| FTS | 0.000 | 0.000 | 0.000 | 0.020 | 0.003 |
| **TRIAXIS** | **0.120** | **0.180** | **0.220** | **0.260** | **0.165** |

**Fusion Lift**: R@1 0.060→0.120 (**+100.0%**) | R@5 0.220→0.220 (**+0.0%**)

---

## Summary — Pipeline Stage Value

| Level | Triaxis R@1 | Triaxis R@5 | Triaxis R@10 | MRR | Stages Active |
|---|:-:|:-:|:-:|:-:|---|
| **simple** | 0.100 | 0.180 | 0.280 | 0.153 | VV+MV+FTS→RRF |
| **scoped** | 0.080 | 0.240 | 0.280 | 0.148 | +Decomposer+Scope |
| **complex** | 0.120 | 0.220 | 0.260 | 0.165 | +Negative+Exclude |

**Scope Filter value**: simple→scoped R@5 0.180→0.240 (**+33.3%**)
**Negative Filter value**: scoped→complex R@5 0.240→0.220 (**-8.3%**)

---

## Per-Query Detail (scoped, top 20)

| File | Query | VV | MV | FTS | Triaxis |
|---|---|:-:|:-:|:-:|:-:|
| dwg02_038.psd | background 중에서 peaceful city street_view | 10 | 7 | - | 2 |
| 34956_nfb09_231A.psd | background 중에서 melancholic winter barren | - | 15 | - | - |
| 48693_VS8_02_369_GENZU.psd | illustration 중에서 overlay ghost silhouett | - | - | - | - |
| 49437_VS9_07_219_GENZU_yuusen.psd | illustration 중에서 lineart overlay charact | - | - | - | - |
| 34996_nfb09_273.psd | illustration 중에서 concept_art mysterious  | - | 20 | - | 7 |
| 35291_nfb09_041.psd | background 중에서 night wooden_wall ghostly | 2 | - | - | - |
| 49827_VS9_03_146_GENZU.psd | illustration 중에서 sci-fi planet stars | - | 11 | - | 8 |
| 50076_VS9_02_321_GENZU.psd | illustration 중에서 abstract space surreal | - | 9 | - | - |
| 48919_VS8_01_314_GENZU.psd | illustration 중에서 overlay festive doors | - | 2 | - | - |
| 47344_VS10_11_025_GENZU.psd | illustration 중에서 glow character pillar | - | 1 | - | 1 |
| 40628_grb03_322.psd | background 중에서 stars night sky serene | - | - | - | - |
| 30663_218_220_222_227_229.psd | illustration 중에서 night stars urban | - | 1 | - | - |
| 47882_VS10_06_005_GENZU.psd | illustration 중에서 night ghost portrait | - | - | - | - |
| 34779_nfb08_037.psd | background 중에서 urban stone_pavement neut | - | 12 | - | 13 |
| 39554_grb13_139.psd | illustration 중에서 symmetry glow fantasy | - | - | - | - |
| 48393_VS10_01_184_GENZU.psd | illustration 중에서 line art portrait drawi | - | - | - | - |
| 39944_grb09_160_BG2.psd | illustration 중에서 castle blue moon night  | 1 | 1 | - | 2 |
| 33075_nfb08_228.psd | illustration 중에서 ghost stars night | - | - | - | - |
| 49441_VS9_07_035_GENZU.psd | illustration 중에서 lineart monochrome urba | - | - | - | - |
| 32397_syt07_098_105_BG2.psd | effect 중에서 glow digital_art teal | - | 1 | - | 4 |

---

## Timing

| Phase | Time |
|---|---|
| init | 0.2s |
| simple_vv | 52.7s |
| simple_mv | 47.7s |
| simple_fts | 0.0s |
| simple_triaxis | 215.8s |
| scoped_vv | 53.9s |
| scoped_mv | 48.2s |
| scoped_fts | 0.1s |
| scoped_triaxis | 245.5s |
| complex_vv | 53.7s |
| complex_mv | 48.5s |
| complex_fts | 0.0s |
| complex_triaxis | 286.7s |
