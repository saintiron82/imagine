"""
WorkerScheduler — central task assignment for Analysis Job System v1.

The server decides which phase (mc/vv/mv) and how many tasks to assign
to each worker, based on:
1. Global queue state (pending counts per phase)
2. Worker capability profile (GPU spec + measured speed)

Workers just say "give me work" — the scheduler decides what.

No legacy dependency (job_queue, manager.py).
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)

# Target batch duration per phase (seconds)
# MC is the bottleneck → longer batch to reduce claim overhead
# VV/MV are fast → shorter batch for responsive progress updates
TARGET_BATCH_SECONDS = {
    "mc": 600,   # ~10 min (MC ~8s/file → ~75 files for fast worker)
    "vv": 120,   # ~2 min  (VV ~0.7s/file → ~170 files)
    "mv": 120,   # ~2 min  (MV ~0.5s/file → ~240 files)
}

# Fallback batch sizes when no measured speed available
DEFAULT_BATCH = {
    "mc": 10,    # conservative: ~1.3 min at 8s/file
    "vv": 50,    # ~35s at 0.7s/file
    "mv": 50,    # ~25s at 0.5s/file
}

# Absolute caps (prevent runaway)
MAX_BATCH = {
    "mc": 100,
    "vv": 500,
    "mv": 500,
}

# MC capability threshold
MC_VRAM_THRESHOLD_GB = 4.0      # Discrete GPU
MC_METAL_THRESHOLD_GB = 16.0    # Apple Metal (unified memory)


class WorkerScheduler:
    """Central scheduler that assigns phase + tasks to workers.

    Usage:
        scheduler = WorkerScheduler(db)
        # Worker connects:
        scheduler.register_worker(session_id, gpu="RTX 4090", vram_gb=24)
        # Worker requests work:
        result = scheduler.assign(session_id)
        # → {"phase": "mc", "count": 20}
        # Worker completes batch:
        scheduler.update_speed(session_id, "mc", files_per_min=7.8)
    """

    def __init__(self, db: SQLiteDB):
        self.db = db
        self._ensure_profile_columns()

    def _ensure_profile_columns(self):
        """Add worker profile columns to worker_sessions if missing."""
        cursor = self.db.conn.cursor()
        columns = {
            "mc_capable": "BOOLEAN DEFAULT 1",
            "mc_speed": "REAL",       # files/min (measured)
            "vv_speed": "REAL",
            "mv_speed": "REAL",
            "gpu_name": "TEXT",
            "vram_gb": "REAL",
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
        is_metal: bool = False,
    ):
        """Register worker GPU specs for initial capability estimation.

        Called on worker connect. Sets mc_capable based on hardware.
        """
        # Estimate MC capability from specs
        if is_metal:
            mc_capable = vram_gb >= MC_METAL_THRESHOLD_GB
        else:
            mc_capable = vram_gb >= MC_VRAM_THRESHOLD_GB

        cursor = self.db.conn.cursor()
        cursor.execute("""
            UPDATE worker_sessions
            SET gpu_name = ?, vram_gb = ?, mc_capable = ?
            WHERE id = ?
        """, (gpu, vram_gb, mc_capable, session_id))
        self.db.conn.commit()

        logger.info(
            f"Scheduler: worker {session_id} registered "
            f"(gpu={gpu}, vram={vram_gb}GB, mc_capable={mc_capable})"
        )

    # ── Speed Updates ───────────────────────────────────────

    def update_speed(
        self, session_id: int, phase: str, files_per_min: float
    ):
        """Update measured processing speed for a worker+phase.

        Called after batch completion with actual elapsed time.
        Uses exponential moving average (alpha=0.3) to smooth noise.
        """
        speed_col = f"{phase}_speed"
        cursor = self.db.conn.cursor()

        # Read current speed for EMA
        try:
            cursor.execute(
                f"SELECT {speed_col} FROM worker_sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            current = row[0] if row and row[0] else None
        except Exception:
            current = None

        if current is not None and current > 0:
            # EMA: 70% old + 30% new
            smoothed = current * 0.7 + files_per_min * 0.3
        else:
            smoothed = files_per_min

        cursor.execute(
            f"UPDATE worker_sessions SET {speed_col} = ? WHERE id = ?",
            (round(smoothed, 2), session_id),
        )
        self.db.conn.commit()

        # If worker proves it can do MC, mark capable
        if phase == "mc" and files_per_min > 0.5:
            cursor.execute(
                "UPDATE worker_sessions SET mc_capable = 1 WHERE id = ?",
                (session_id,),
            )
            self.db.conn.commit()

    # ── Central Assignment ──────────────────────────────────

    def assign(self, session_id: int) -> Dict[str, Any]:
        """Decide which phase and how many tasks to assign to a worker.

        Returns:
            {"phase": "mc"|"vv"|"mv"|None, "count": int}
            phase=None means no work available.
        """
        cursor = self.db.conn.cursor()

        # 1. Get pending counts per phase
        pending = self._get_pending_counts(cursor)
        mc_p, vv_p, mv_p = pending["mc"], pending["vv"], pending["mv"]

        if mc_p + vv_p + mv_p == 0:
            return {"phase": None, "count": 0}

        # 2. Get worker profile
        profile = self._get_worker_profile(cursor, session_id)

        # 3. Assignment decision
        phase, count = self._decide(profile, pending, cursor)

        return {"phase": phase, "count": count}

    def _batch_for_phase(self, phase: str, profile: dict, pending: int) -> int:
        """Calculate batch size: target_time × worker_speed.

        Large batches minimize model switch overhead.
        """
        speed_key = f"{phase}_speed"
        speed = profile.get(speed_key)  # files/min or None

        if speed and speed > 0:
            target = TARGET_BATCH_SECONDS[phase]
            batch = int(target * speed / 60.0)
        else:
            batch = DEFAULT_BATCH[phase]

        batch = max(1, min(batch, pending, MAX_BATCH[phase]))
        return batch

    def _pick_vv_or_mv(self, profile: dict, vv_p: int, mv_p: int) -> tuple:
        """Pick VV or MV based on pending count."""
        if vv_p >= mv_p and vv_p > 0:
            return ("vv", self._batch_for_phase("vv", profile, vv_p))
        if mv_p > 0:
            return ("mv", self._batch_for_phase("mv", profile, mv_p))
        return (None, 0)

    def _decide(
        self, profile: dict, pending: dict, cursor
    ) -> tuple:
        """Core scheduling algorithm — pool-aware, phase-sticky.

        Principles:
        1. PHASE STICKY: keep same phase while pending > 0 (minimize model switches)
        2. MC PRIORITY: fastest MC workers get MC; slow ones do VV/MV
        3. MC EXCLUSION: if worker MC speed < 30% of best, skip MC
        4. COLD START: no measurement → small trial batch (3 files)

        Returns: (phase, count)
        """
        mc_p = pending["mc"]
        vv_p = pending["vv"]
        mv_p = pending["mv"]
        mc_capable = profile.get("mc_capable", True)
        current_phase = profile.get("current_phase")

        # ── Rule 0: Phase sticky — same phase while pending remains ──
        if current_phase and pending.get(current_phase, 0) > 0:
            # Verify worker can still do this phase
            if current_phase == "mc" and not mc_capable:
                pass  # fall through to reassignment
            else:
                count = self._batch_for_phase(current_phase, profile, pending[current_phase])
                return (current_phase, count)

        # ── Rule 1: MC not capable → VV/MV only ──
        if not mc_capable:
            return self._pick_vv_or_mv(profile, vv_p, mv_p)

        # ── Rule 2: No MC pending → VV/MV ──
        if mc_p == 0:
            return self._pick_vv_or_mv(profile, vv_p, mv_p)

        # ── Rule 3: MC pending — should THIS worker do MC? ──
        my_mc_speed = profile.get("mc_speed")

        # 3a. Cold start: no measurement → trial batch (3 files)
        if my_mc_speed is None:
            return ("mc", min(3, mc_p))

        # 3b. Check if better MC workers exist
        cursor.execute("""
            SELECT mc_speed FROM worker_sessions
            WHERE status = 'online' AND mc_capable = 1 AND mc_speed IS NOT NULL
            ORDER BY mc_speed DESC
        """)
        all_mc_speeds = [r[0] for r in cursor.fetchall() if r[0] and r[0] > 0]

        if all_mc_speeds:
            best_mc = all_mc_speeds[0]

            # If my speed < 30% of best AND there's VV/MV work → skip MC
            if my_mc_speed < best_mc * 0.3 and (vv_p > 0 or mv_p > 0):
                return self._pick_vv_or_mv(profile, vv_p, mv_p)

        # ── Rule 4: MC assigned ──
        count = self._batch_for_phase("mc", profile, mc_p)
        return ("mc", count)

    # ── Helpers ─────────────────────────────────────────────

    def _get_pending_counts(self, cursor) -> Dict[str, int]:
        """Query pending task counts per AI phase (mc/vv/mv)."""
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
        """Read worker capability profile from DB."""
        cursor.execute("""
            SELECT mc_capable, mc_speed, vv_speed, mv_speed,
                   gpu_name, vram_gb, current_phase
            FROM worker_sessions WHERE id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if not row:
            return {"mc_capable": True}
        return {
            "mc_capable": bool(row[0]) if row[0] is not None else True,
            "mc_speed": row[1],
            "vv_speed": row[2],
            "mv_speed": row[3],
            "gpu_name": row[4],
            "vram_gb": row[5],
            "current_phase": row[6],
        }

    def get_status(self) -> dict:
        """Get scheduler status for monitoring/debugging."""
        cursor = self.db.conn.cursor()
        pending = self._get_pending_counts(cursor)

        # Count online workers by capability
        cursor.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN mc_capable = 1 THEN 1 ELSE 0 END) AS mc_capable
            FROM worker_sessions
            WHERE status = 'online'
        """)
        row = cursor.fetchone()
        total_workers = row[0] or 0
        mc_workers = row[1] or 0

        return {
            "pending": pending,
            "workers": {
                "total": total_workers,
                "mc_capable": mc_workers,
                "vv_mv_only": total_workers - mc_workers,
            },
        }
