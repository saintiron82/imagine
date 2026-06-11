"""
WorkerScheduler — central task assignment for Analysis Job System v1.

Implements Algorithm E' (Pressure-based scheduling):
- GPU class detection (strong/weak/cpu)
- Per-phase pressure = (pending / (workers_on + 1)) × phase_weight
- MC penalty by GPU class (cpu can't, weak penalized)
- Phase stability (don't switch if current phase has work — minimize model load)
- MV completion bonus (finish jobs faster)
- Dynamic batch sizing (target time × measured speed)

Ported from legacy manager.py._pick_best_phase() + _decide_worker_mode().
No retired queue-manager dependency.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────

# Per-phase processing time (seconds/file) — used for pressure weighting
PHASE_TIME = {"mc": 8, "vv": 0.5, "mv": 0.25}

# MC penalty by GPU class — higher = less likely to get MC
# None = cannot do MC at all
MC_PENALTY = {
    "strong": 1.0,     # Full speed MC
    "weak": 2.0,       # Can do MC but slower → penalized
    # "cpu": None      # Cannot do MC (not in dict → excluded)
}

# Target batch duration (seconds) — larger = fewer model switches
TARGET_BATCH_SECONDS = {
    "mc": 600,   # ~10 min
    "vv": 120,   # ~2 min
    "mv": 120,   # ~2 min
}

# Fallback batch sizes (no measured speed yet)
DEFAULT_BATCH = {"mc": 10, "vv": 50, "mv": 50}

# Absolute caps
MAX_BATCH = {"mc": 200, "vv": 500, "mv": 500}

# Cold start trial batch (first claim, no measurements)
COLD_START_BATCH = 3

# Phase stability threshold: only switch if best > current × this factor
STABILITY_FACTOR = 2.0


class WorkerScheduler:
    """Central scheduler — decides phase + batch size for each worker.

    Algorithm E' (pressure-based):
    1. Classify worker GPU (strong/weak/cpu)
    2. Calculate pressure per phase: (pending / (workers_on + 1)) × time_weight
    3. Apply MC penalty by GPU class
    4. Phase sticky: stay in current phase unless another has 2× pressure
    5. MV completion bonus: finishing MV = file complete → prioritize
    6. Batch size: target_time × measured_speed (dynamic per worker)
    """

    def __init__(self, db: SQLiteDB):
        self.db = db
        self._ensure_profile_columns()

    def _ensure_profile_columns(self):
        """Add worker profile columns to worker_sessions if missing."""
        cursor = self.db.conn.cursor()
        columns = {
            "mc_capable": "BOOLEAN DEFAULT 1",
            "mc_speed": "REAL",
            "vv_speed": "REAL",
            "mv_speed": "REAL",
            "gpu_name": "TEXT",
            "vram_gb": "REAL",
            "gpu_class": "TEXT",   # strong/weak/cpu
        }
        for col, typedef in columns.items():
            try:
                cursor.execute(f"SELECT {col} FROM worker_sessions LIMIT 1")
            except Exception:
                try:
                    cursor.execute(
                        f"ALTER TABLE worker_sessions ADD COLUMN {col} {typedef}"
                    )
                    self.db.conn.commit()
                except Exception:
                    pass

    # ── Worker Registration ─────────────────────────────────

    def register_worker(
        self, session_id: int, gpu: str = "", vram_gb: float = 0,
        is_metal: bool = False, scores: dict = None,
    ):
        """Register worker with benchmark scores.

        If scores provided (from worker_profile.json):
          - Use measured speeds directly
          - Determine MC capability from actual measurement (speed > 0)
          - GPU class from scores grade

        If no scores (cold start):
          - Fall back to GPU spec estimation
        """
        cursor = self.db.conn.cursor()

        if scores and scores.get("phases"):
            # Score-based registration — actual measurements
            phases = scores["phases"]
            mc_phase = phases.get("mc", {})
            mc_capable = mc_phase.get("status") == "ok" and mc_phase.get("speed", 0) > 0

            # GPU class from grade
            grade = scores.get("grade", "F")
            gpu_class = {"S": "strong", "A": "strong", "B": "weak", "C": "weak", "F": "cpu"}.get(grade, "cpu")

            # Apply measured speeds
            mc_speed = phases.get("mc", {}).get("speed")
            vv_speed = phases.get("vv", {}).get("speed")
            mv_speed = phases.get("mv", {}).get("speed")

            cursor.execute("""
                UPDATE worker_sessions
                SET gpu_name = ?, vram_gb = ?, gpu_class = ?, mc_capable = ?,
                    mc_speed = ?, vv_speed = ?, mv_speed = ?
                WHERE id = ?
            """, (gpu, vram_gb, gpu_class, mc_capable,
                  mc_speed, vv_speed, mv_speed, session_id))
            self.db.conn.commit()

            incapable = scores.get("incapable", [])
            total = scores.get("total", 0)
            logger.info(
                f"Scheduler: worker {session_id} registered with scores "
                f"(grade={grade}, total={total}, mc={mc_speed}/m, "
                f"vv={vv_speed}/m, mv={mv_speed}/m, "
                f"incapable={incapable or 'none'})"
            )
        else:
            # Spec-based fallback (no benchmark data)
            gpu_class = self._classify_gpu(gpu, vram_gb, is_metal)
            mc_capable = gpu_class in MC_PENALTY

            cursor.execute("""
                UPDATE worker_sessions
                SET gpu_name = ?, vram_gb = ?, gpu_class = ?, mc_capable = ?
                WHERE id = ?
            """, (gpu, vram_gb, gpu_class, mc_capable, session_id))
            self.db.conn.commit()

            logger.info(
                f"Scheduler: worker {session_id} registered (spec-based) "
                f"(gpu={gpu}, vram={vram_gb}GB, class={gpu_class}, mc={mc_capable})"
            )

    def _classify_gpu(self, gpu: str, vram_gb: float, is_metal: bool) -> str:
        """Classify GPU into: strong, weak, cpu."""
        if not gpu and vram_gb == 0:
            return "cpu"

        # Read active tier's VRAM requirement
        try:
            from backend.utils.config import get_config
            cfg = get_config()
            tiers = cfg.get("ai_mode.tiers", {})
            active_tier = cfg.get("ai_mode.override") or "pro"
            tier_cfg = tiers.get(active_tier, {})
            vram_min = tier_cfg.get("vram_min", 4)
        except Exception:
            vram_min = 4

        if is_metal:
            # Apple Metal: unified memory, needs more
            return "strong" if vram_gb >= 16 else ("weak" if vram_gb >= 8 else "cpu")
        else:
            # Discrete GPU
            threshold = vram_min * 0.9
            if vram_gb >= threshold:
                return "strong"
            elif vram_gb > 0:
                return "weak"
            return "cpu"

    # ── Speed Updates ───────────────────────────────────────

    def update_speed(self, session_id: int, phase: str, files_per_min: float):
        """Update measured speed (EMA smoothed)."""
        speed_col = f"{phase}_speed"
        cursor = self.db.conn.cursor()

        try:
            cursor.execute(
                f"SELECT {speed_col} FROM worker_sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            current = row[0] if row and row[0] else None
        except Exception:
            current = None

        if current and current > 0:
            smoothed = current * 0.7 + files_per_min * 0.3
        else:
            smoothed = files_per_min

        cursor.execute(
            f"UPDATE worker_sessions SET {speed_col} = ? WHERE id = ?",
            (round(smoothed, 2), session_id),
        )
        self.db.conn.commit()

    # ── Central Assignment ──────────────────────────────────

    def assign(self, session_id: int) -> Dict[str, Any]:
        """Decide phase + batch size for a worker.

        Returns: {"phase": "mc"|"vv"|"mv"|None, "count": int}
        """
        cursor = self.db.conn.cursor()

        # 1. Pending counts (exclude paused phases)
        pending = self._get_pending_counts(cursor)
        try:
            from backend.server.routers.analysis import get_paused_phases
            paused = get_paused_phases(self.db)
            if paused.get("mc"): pending["mc"] = 0
            if paused.get("vv"): pending["vv"] = 0
            if paused.get("mv"): pending["mv"] = 0
        except Exception:
            pass
        mc_p, vv_p, mv_p = pending["mc"], pending["vv"], pending["mv"]
        if mc_p + vv_p + mv_p == 0:
            return {"phase": None, "count": 0}

        # 2. Worker profile
        profile = self._get_worker_profile(cursor, session_id)
        gpu_class = profile.get("gpu_class", "cpu")
        current_phase = profile.get("current_phase")

        # 3. Throttle check
        throttle = profile.get("throttle", "normal")
        if throttle == "critical":
            return {"phase": None, "count": 0}

        # 4. Workers currently assigned to each phase
        cursor.execute("""
            SELECT assigned_mode, COUNT(*) FROM worker_sessions
            WHERE status = 'online' AND assigned_mode IS NOT NULL
            GROUP BY assigned_mode
        """)
        workers_on = dict(cursor.fetchall())

        # 5. Determine capable phases
        # All workers can do VV and MV
        capable = {"vv": vv_p, "mv": mv_p}
        # MC capability: measured speed > 0, or GPU class allows it
        mc_speed = profile.get("mc_speed")
        if mc_speed is not None and mc_speed > 0:
            # Proven capable (benchmark or runtime measurement)
            capable["mc"] = mc_p
        elif mc_speed is None and gpu_class in MC_PENALTY:
            # No measurement yet, but GPU class suggests capable → allow (trial)
            capable["mc"] = mc_p
        # mc_speed == 0 → tried and failed → MC excluded

        # 5.5 Admin pin: restrict this worker to the pinned phase.
        # ("mc_only" is a legacy alias for "mc"; "full"/None = no pin)
        pin = {"mc_only": "mc", "full": None}.get(
            profile.get("mode_override"), profile.get("mode_override"))
        if pin in ("mc", "vv", "mv"):
            capable = {k: v for k, v in capable.items() if k == pin}
            if not capable:
                # pinned to a phase this worker cannot do (or no pending)
                return {"phase": None, "count": 0}

        # 6. Algorithm E' — pressure × speed_factor selection
        phase = self._pick_best_phase(
            capable, workers_on, current_phase, profile
        )

        if not phase:
            return {"phase": None, "count": 0}

        # 7. Batch size
        count = self._batch_for_phase(phase, profile, pending.get(phase, 0))

        # 8. Cold start: no speed measurement → small trial
        speed_key = f"{phase}_speed"
        if profile.get(speed_key) is None:
            count = min(count, COLD_START_BATCH)

        # 9. Throttle reduction
        if throttle == "danger":
            count = min(count, 1)
        elif throttle == "warning":
            count = max(1, count // 2)

        # 10. Update assigned_mode for monitoring
        if phase != current_phase:
            cursor.execute(
                "UPDATE worker_sessions SET assigned_mode = ? WHERE id = ?",
                (phase, session_id),
            )
            self.db.conn.commit()

        return {"phase": phase, "count": count}

    # Design speed baselines (files/min) — used to normalize speed_factor
    SPEED_BASELINE = {"mc": 8, "vv": 80, "mv": 120}

    def _speed_factor(self, profile: dict, phase: str) -> float:
        """How much does this worker contribute to this phase?

        Measured speed → normalize against baseline.
        No measurement → fall back to GPU class estimation.

        Returns: >1 = faster than baseline, <1 = slower, 0 = cannot do this phase.
        """
        speed = profile.get(f"{phase}_speed")
        if speed and speed > 0:
            return speed / self.SPEED_BASELINE.get(phase, 1)

        # No measurement: GPU class based estimation
        gpu_class = profile.get("gpu_class", "cpu")
        if phase == "mc":
            penalty = MC_PENALTY.get(gpu_class)
            return (1.0 / penalty) if penalty else 0  # cpu → 0
        return 1.0  # VV/MV: all workers can do, assume baseline

    def _pick_best_phase(
        self, claimable: Dict[str, int], workers_on: Dict[str, int],
        current_phase: Optional[str], profile: dict,
    ) -> Optional[str]:
        """Algorithm E' — pressure × speed_factor phase selection.

        pressure = (pending / (workers+1)) × phase_time × speed_factor

        speed_factor uses MEASURED speed when available (from benchmark or
        runtime). Falls back to GPU class estimation for unmeasured workers.

        Factors:
        - speed_factor: this worker's actual throughput for this phase
        - MV completion bonus (+10 per pending, finishing = file complete)
        - Unserved phase boost (×1.5 if no worker on this phase)
        - Phase stability (stay in current unless better has 2× pressure)
        """
        pressure = {}

        for phase, pending in claimable.items():
            if pending <= 0:
                continue

            # Speed factor: how fast is THIS worker at THIS phase?
            sf = self._speed_factor(profile, phase)
            if sf <= 0:
                continue  # Cannot do this phase (e.g., CPU + MC)

            # Base pressure: pending work per available worker, weighted by time
            w = PHASE_TIME.get(phase, 1)
            n = workers_on.get(phase, 0)
            p = (pending / (n + 1)) * w * sf

            # Unserved phase boost: no worker assigned → 1.5× priority
            if n == 0 and pending > 0:
                p *= 1.5

            # MV completion bonus: each MV done = 1 file fully complete.
            # Capped — an unbounded pending×10 bonus let any large MV backlog
            # outweigh MC pressure for every worker (strong GPUs included),
            # causing cluster-wide mode flapping. The cap keeps the intent
            # ("drain MVs when otherwise comparable / near the MC tail")
            # without overriding a large MC backlog.
            if phase == "mv":
                p += min(pending, 50) * 10

            pressure[phase] = p

        if not pressure:
            return None

        best = max(pressure, key=pressure.get)

        # Phase stability: only switch if best > current × STABILITY_FACTOR
        if current_phase and current_phase in pressure and pressure[current_phase] > 0:
            if pressure[best] <= pressure[current_phase] * STABILITY_FACTOR:
                return current_phase

        return best if pressure[best] > 0 else None

    def _batch_for_phase(self, phase: str, profile: dict, pending: int) -> int:
        """Batch size = target_time × measured_speed."""
        speed = profile.get(f"{phase}_speed")

        if speed and speed > 0:
            target = TARGET_BATCH_SECONDS[phase]
            batch = int(target * speed / 60.0)
        else:
            batch = DEFAULT_BATCH[phase]

        return max(1, min(batch, pending, MAX_BATCH[phase]))

    # ── Helpers ─────────────────────────────────────────────

    def _get_pending_counts(self, cursor) -> Dict[str, int]:
        """Pending task counts per AI phase."""
        cursor.execute("""
            SELECT
                SUM(CASE WHEN parse_status='done' AND mc_status='pending'
                    THEN 1 ELSE 0 END) AS mc,
                SUM(CASE WHEN parse_status='done' AND vv_status='pending'
                    THEN 1 ELSE 0 END) AS vv,
                SUM(CASE WHEN mc_status='done' AND mv_status='pending'
                    THEN 1 ELSE 0 END) AS mv
            FROM file_tasks ft
            JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
            WHERE aj.status = 'active'
        """)
        row = cursor.fetchone()
        return {
            "mc": row[0] or 0,
            "vv": row[1] or 0,
            "mv": row[2] or 0,
        }

    def _get_worker_profile(self, cursor, session_id: int) -> dict:
        """Read worker profile from DB."""
        cursor.execute("""
            SELECT mc_capable, mc_speed, vv_speed, mv_speed,
                   gpu_name, vram_gb, gpu_class, current_phase,
                   resources_json, processing_mode_override
            FROM worker_sessions WHERE id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if not row:
            return {"mc_capable": False, "gpu_class": "cpu"}

        throttle = "normal"
        if row[8]:
            try:
                res = json.loads(row[8])
                throttle = res.get("throttle_level", "normal")
            except Exception:
                pass

        return {
            "mc_capable": bool(row[0]) if row[0] is not None else False,
            "mc_speed": row[1],
            "vv_speed": row[2],
            "mv_speed": row[3],
            "gpu_name": row[4],
            "vram_gb": row[5],
            "gpu_class": row[6] or "cpu",
            "current_phase": row[7],
            "throttle": throttle,
            "mode_override": row[9],
        }

    def get_status(self) -> dict:
        """Scheduler status for monitoring."""
        cursor = self.db.conn.cursor()
        pending = self._get_pending_counts(cursor)

        cursor.execute("""
            SELECT gpu_class, COUNT(*) FROM worker_sessions
            WHERE status = 'online'
            GROUP BY gpu_class
        """)
        by_class = dict(cursor.fetchall())

        cursor.execute("""
            SELECT assigned_mode, COUNT(*) FROM worker_sessions
            WHERE status = 'online' AND assigned_mode IS NOT NULL
            GROUP BY assigned_mode
        """)
        by_mode = dict(cursor.fetchall())

        return {
            "pending": pending,
            "workers_by_class": by_class,
            "workers_by_mode": by_mode,
        }
