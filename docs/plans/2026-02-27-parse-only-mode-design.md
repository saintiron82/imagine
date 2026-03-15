# Design: `parse_only` Global Worker Mode

**Date**: 2026-02-27
**Status**: Approved

## Summary

Add a third global worker mode `parse_only` where the server only performs Phase P (parsing/thumbnails) and delegates all GPU-intensive work (V/VV/MV) to workers. The server loads zero GPU models.

## Current Mode System

| Global Mode | Server Role | Worker Role |
|-------------|------------|-------------|
| **auto** | P + gap-fill V (situational) | full/embed_only (auto-detect) |
| **mc_only** | P + VV + MV | V (MC only) |

## New Mode

| Global Mode | Server Role | Worker Role |
|-------------|------------|-------------|
| **parse_only** | P only (zero GPU) | full (V + VV + MV) |

## Data Flow

```
[Server ParseAheadPool]              [Worker (full mode)]
Phase P: Parse + thumbnail
  -> job_queue.parse_status='parsed'
  -> queue up                         claim(N) ->
                                      Phase V: VLM -> MC
                                      Phase VV: SigLIP2 -> VV
                                      Phase MV: Qwen3-Embedding -> MV
                                      -> complete_job() upload
```

## Server Behavior (ParseAheadPool)

When `_processing_mode == "parse_only"`:
- `_run_pre_parse_buffer()`: Phase P only, no VV encoding (unlike mc_only)
- VLM, SigLIP2, Qwen3-Embedding: **never loaded**
- Gap-fill V: **disabled**
- EmbedAheadPool: **not started**
- No workers online: Parse continues, parsed jobs queue up waiting for workers

## Worker Behavior

- Receives `processing_mode = "full"` (identical to existing full mode)
- Phase P: skipped (server pre-parsed)
- Processes V -> VV -> MV in batch

## Mode Transition Logic (_recalculate_server_pools)

```
global_mode == "parse_only":
  -> ParseAheadPool._processing_mode = "parse_only"
  -> EmbedAheadPool: stop
  -> Worker processing_mode response: "full"
  -> No fallback to auto when workers offline (parse and wait)
```

## Admin UI

Three-button selector (existing two + new):

```
[MC Only] [Parse Only] [Auto]
```

Parse Only description: "Server handles Parse only, V+VV+MV fully delegated to workers"

## Files to Modify

| File | Change |
|------|--------|
| `backend/server/queue/manager.py` | Allow `"parse_only"` in `get_processing_mode()` |
| `backend/server/queue/parse_ahead.py` | Add `parse_only` branch in main loop |
| `backend/server/routers/workers.py` | `_recalculate_server_pools`, global config API, connect/heartbeat responses |
| `frontend/src/pages/AdminPage.jsx` | Third mode button |
| `frontend/src/components/ClientWorkerView.jsx` | `parse_only` mode Phase display |
| `frontend/src/i18n/locales/en-US.json` | English labels |
| `frontend/src/i18n/locales/ko-KR.json` | Korean labels |

## Edge Cases

- **No workers online**: Server keeps parsing, jobs queue up. No auto fallback.
- **Worker disconnects mid-batch**: Standard retry mechanism (job timeout -> re-queue).
- **Mode switch from parse_only to mc_only/auto**: ParseAheadPool mode updates, server may start loading models as needed.
- **Throughput measurement**: Uses `completed_at` (same as non-mc_only modes) since workers complete the full pipeline.
