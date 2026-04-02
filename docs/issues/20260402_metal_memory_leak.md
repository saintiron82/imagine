# Bug Report: Metal GPU 메모리 미해제 (IOAccelerator 누적)

| 항목 | 내용 |
|------|------|
| **보고일** | 2026-04-02 |
| **보고자** | saintiron |
| **발견 경로** | Qwen3.5 전환 후 시스템 메모리 94% 사용 확인 → footprint 분석 |
| **환경** | Apple M5 32GB, macOS 26.2, MLX + Metal 4, Electron 앱 |
| **심각도** | **높음** — 장시간 운영 시 시스템 전체 메모리 고갈 |
| **상태** | Open |

---

## BUG-001: Metal GPU 버퍼(IOAccelerator) 누적 미해제

| 항목 | 내용 |
|------|------|
| **심각도** | **높음** |
| **재현** | 서버 장시간 운영 + 모델 로드/언로드 반복 (MC→VV→MV Phase 전환, 또는 테스트 스크립트 실행) |
| **현상** | `footprint -p <PID>` 결과 Imagine-Pipeline 프로세스가 IOAccelerator(graphics) **23GB** 점유. 프로세스 RSS는 480MB에 불과하나 Metal 버퍼가 커널 레벨에서 해제되지 않고 누적 |
| **측정 데이터** | |

```
footprint -p 53525:
  IOAccelerator (graphics):                    23 GB  ← Metal GPU 버퍼
  Owned physical footprint (unmapped) (graphics): 3.3 GB
  MALLOC_SMALL:                                756 MB
  프로세스 RSS:                                 480 MB
  
시스템 전체:
  Total: 32 GB, Used: 23.4 GB (79.9%), Available: 6.4 GB
  GPU Pageable allocation: 29.2 GB
```

| 항목 | 내용 |
|------|------|
| **원인** | MLX/Metal의 GPU 버퍼 할당이 `mx.clear_cache()` / `torch.mps.empty_cache()` / `gc.collect()` 호출 후에도 IOAccelerator 레벨에서 완전히 반환되지 않음. Phase별 모델 교체(VLM→SigLIP2→Qwen3-Embedding) 반복 시 누적 |
| **영향** | 장시간 운영 시 시스템 Available 메모리 감소 → swap 발생 → 전체 성능 저하. 32GB 시스템에서 23GB를 GPU 버퍼가 점유하면 OS + 다른 앱에 여유 없음 |
| **임시 해결** | 서버 재시작 (프로세스 종료 시 IOAccelerator 버퍼 전부 해제) |

### 근본 해결 방향

| 방안 | 설명 | 난이도 |
|------|------|--------|
| **A. 모델 상주 유지** | Phase별 교체 대신 모든 모델을 동시에 메모리에 유지. 로드/언로드 반복 제거. 9B(6.6GB) + SigLIP2(~1GB) + Embedding(~1GB) = ~9GB로 32GB에서 가능 | 중간 |
| **B. 프로세스 격리** | VLM/VV/MV를 별도 subprocess로 실행. Phase 완료 후 subprocess 종료 → OS가 IOAccelerator 강제 회수 | 높음 |
| **C. MLX 메모리 관리 개선** | `mx.clear_cache()` 외에 `mx.metal.reset_peak_memory()`, Metal 버퍼 명시적 해제 API 조사. mlx-vlm 0.4.2+ 업그레이드로 개선 여부 확인 | 낮음 |
| **D. 주기적 재시작** | N시간 또는 N배치마다 서버 자동 재시작. 가장 단순하지만 사용자 경험 저하 | 낮음 |

### 관련 파일

| 파일 | 역할 |
|------|------|
| `backend/worker/worker_daemon.py` | `_unload_vlm()`, `_unload_vv()`, `_unload_mv()` — 모델 언로드 + `gc.collect()` + `torch.mps.empty_cache()` |
| `backend/vision/mlx_adapter.py` | MLX VLM 모델 로드/해제 |
| `backend/vector/siglip2_encoder.py` | SigLIP2 VV 인코더 (MPS) |
| `backend/vector/text_embedding.py` | Qwen3-Embedding MV 인코더 (transformers) |
| `backend/server/embedded_worker.py:137-146` | 배치 간 GPU cleanup (`torch.mps.empty_cache()`) |

### 검증 방법

```bash
# 서버 재시작 전
footprint -p <Imagine-Pipeline PID> | head -20

# 서버 재시작 후 (메모리 해제 확인)
footprint -p <new PID> | head -20

# 배치 N회 후 누적 확인
watch -n 60 'footprint -p <PID> | grep IOAccelerator'
```

---

## 우선순위

| 순위 | 방안 | ROI |
|:----:|------|-----|
| 1 | **C. MLX 메모리 관리** — 가장 적은 변경으로 효과 확인 | 높음 |
| 2 | **A. 모델 상주** — Phase별 언로드 제거, 메모리 여유 있으면 최선 | 중간 |
| 3 | **D. 주기적 재시작** — 즉시 적용 가능한 안전망 | 낮음 |
| 4 | **B. 프로세스 격리** — 아키텍처 변경 필요, 장기 과제 | 낮음 |
