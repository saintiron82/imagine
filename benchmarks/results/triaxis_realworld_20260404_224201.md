# Triaxis Real-World Search Benchmark

**Date**: 2026-04-04 22:42
**Queries**: 50 (natural Korean, from MC data)
**DB Size**: 11174 images

## Query Examples (actual generated)

| Level | Example |
|---|---|
| **content** | `벽이 있는 tileable 이미지` |
| **folder** | `벽이 있는 tileable 이미지` |
| **complex** | `벽이 있는 tileable 이미지` |

---

## CONTENT

| Axis | R@1 | R@3 | R@5 | R@10 | MRR |
|---|:-:|:-:|:-:|:-:|:-:|
| VV | 0.060 | 0.080 | 0.100 | 0.100 | 0.079 |
| MV | 0.040 | 0.100 | 0.120 | 0.160 | 0.079 |
| FTS | 0.040 | 0.040 | 0.040 | 0.060 | 0.043 |
| **TRIAXIS** | **0.140** | **0.260** | **0.260** | **0.340** | **0.203** |

**Fusion Lift R@5**: 0.120 → 0.260 (**+116.7%**)

---

## FOLDER

| Axis | R@1 | R@3 | R@5 | R@10 | MRR |
|---|:-:|:-:|:-:|:-:|:-:|
| VV | 0.080 | 0.100 | 0.120 | 0.120 | 0.098 |
| MV | 0.020 | 0.100 | 0.100 | 0.120 | 0.058 |
| FTS | 0.040 | 0.060 | 0.060 | 0.080 | 0.050 |
| **TRIAXIS** | **0.200** | **0.300** | **0.340** | **0.420** | **0.266** |

**Fusion Lift R@5**: 0.120 → 0.340 (**+183.3%**)

---

## COMPLEX

| Axis | R@1 | R@3 | R@5 | R@10 | MRR |
|---|:-:|:-:|:-:|:-:|:-:|
| VV | 0.060 | 0.120 | 0.140 | 0.160 | 0.099 |
| MV | 0.040 | 0.120 | 0.120 | 0.140 | 0.077 |
| FTS | 0.040 | 0.040 | 0.040 | 0.060 | 0.043 |
| **TRIAXIS** | **0.100** | **0.200** | **0.220** | **0.300** | **0.156** |

**Fusion Lift R@5**: 0.140 → 0.220 (**+57.1%**)

---

## Summary

| Level | Triaxis R@1 | R@5 | R@10 | MRR | Best Single R@5 | Lift |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **content** | 0.140 | 0.260 | 0.340 | 0.203 | 0.120 | **+116.7%** |
| **folder** | 0.200 | 0.340 | 0.420 | 0.266 | 0.120 | **+183.3%** |
| **complex** | 0.100 | 0.220 | 0.300 | 0.156 | 0.140 | **+57.1%** |

---

## Per-Query Detail (folder, top 20)

| File | Query | VV | MV | FTS | Triaxis |
|---|---|:-:|:-:|:-:|:-:|
| 46626_112+steel2_0204f.png | dark 밤 찾아줘 | - | - | - | 2 |
| 34514_nfb04_013.psd | 벽이 있는 tileable 이미지 | 11 | - | - | 8 |
| dwg01_206_BG4_BGonly.psd | 도시 낮폴더 clear sky traditional architectur | - | - | - | - |
| 34122_nfb09_111.psd | apothecary과 indoor 배경 | - | 3 | - | 6 |
| 46402_monooki_sozai_6.psd | wood brown | 15 | 2 | - | 3 |
| 40193_grb07_136.psd | starry sky이 있는 aurora 이미지 | - | - | - | - |
| c-131.psd | 실내소품 중에 red_curtain indoor 찾아줘 | - | - | 17 | - |
| 32303_syt07_009.psd | #07폴더 잔디 숲 | - | - | - | 11 |
| 47820_VS10_07_139_genzu.psd | #07폴더에서 불이 있는 VS10_07_139 이미지 | - | - | 1 | 1 |
| 40364_grb05_067.psd | purple이 있는 cave 이미지 | - | - | - | 9 |
| 40120_grb10_135.psd | blue tones과 하늘 배경 | - | - | - | - |
| BR01_032.psd | 레이다스폴더에서 crystals이 있는 bioluminescent 이미지 | - | 9 | - | 17 |
| 50106_VS9_02_288_GENZU.psd | #02폴더 VS9_02_288 galaxy | - | 1 | 1 | 1 |
| 38157_켄신-232.JPG | shoji_screen과 tatami_floor 배경 | - | - | - | - |
| 32381_syt07_086_197_BG1.psd | #07에서 peaceful과 overgrown 있는 이미지 | - | - | - | - |
| dwg05_253.psd | 도시 낮 중에 portrait lineart 찾아줘 | - | - | - | - |
| 41198_grb02_107.psd | mysterious warm lights 찾아줘 | - | - | - | - |
| 46391_kyoukai_choukoku.psd | Korean과 sculpture 배경 | 1 | 2 | - | 1 |
| 40309_grb07_312.psd | dark sky과 orange glow 배경 | - | - | - | - |
| 41517_grb13_293_303_BG1.psd | gold architecture altar | 5 | - | - | 13 |

---

## Timing

| Phase | Time |
|---|---|
| init | 0.2s |
| content_vv | 56.4s |
| content_mv | 50.9s |
| content_fts | 0.0s |
| content_triaxis | 223.8s |
| folder_vv | 53.3s |
| folder_mv | 48.1s |
| folder_fts | 0.0s |
| folder_triaxis | 166.0s |
| complex_vv | 53.2s |
| complex_mv | 48.4s |
| complex_fts | 0.0s |
| complex_triaxis | 199.7s |
