"""
Schedule Parser — determines if the worker should be active based on
time-of-day, day-of-week, and timezone settings from user-settings.yaml.

Configuration (user-settings.yaml):

    worker:
      schedule:
        active_hours: "09:00-23:00"    # Active time range (omit = 24h)
        active_days: [1,2,3,4,5]       # Active weekdays 1=Mon..7=Sun (omit = every day)
        timezone: "Asia/Seoul"          # IANA timezone (omit = system local)

When no schedule section exists, the worker is always active (24/7).
"""

import logging
from datetime import datetime, time as dt_time
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_time_range(raw: str) -> Optional[tuple]:
    """Parse 'HH:MM-HH:MM' into (start_time, end_time) tuple.

    Returns None if the format is invalid.
    Supports overnight ranges like '22:00-06:00'.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if "-" not in raw:
        return None

    parts = raw.split("-", 1)
    if len(parts) != 2:
        return None

    try:
        start_parts = parts[0].strip().split(":")
        end_parts = parts[1].strip().split(":")
        start = dt_time(int(start_parts[0]), int(start_parts[1]))
        end = dt_time(int(end_parts[0]), int(end_parts[1]))
        return (start, end)
    except (ValueError, IndexError) as e:
        logger.warning(f"Invalid active_hours format '{raw}': {e}")
        return None


def _get_schedule_config() -> dict:
    """Load schedule config from user-settings.yaml via AppConfig.

    Returns the worker.schedule section dict, or empty dict if absent.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        worker = cfg.get("worker", {})
        if isinstance(worker, dict):
            return worker.get("schedule", {}) or {}
        # AppConfig.get returns the value directly for dotted keys
        schedule = cfg.get("worker.schedule", {})
        return schedule if isinstance(schedule, dict) else {}
    except Exception as e:
        logger.debug(f"Failed to load schedule config: {e}")
        return {}


def is_active_now(schedule_config: Optional[dict] = None) -> bool:
    """Determine if the current time falls within the configured active window.

    Args:
        schedule_config: Schedule configuration dict. If None, reads from
            user-settings.yaml automatically. Expected keys:
            - active_hours: "HH:MM-HH:MM" (optional, omit = always active)
            - active_days: list of ints 1-7, 1=Mon..7=Sun (optional, omit = every day)
            - timezone: IANA timezone string (optional, omit = system local)

    Returns:
        True if the worker should be active now, False otherwise.
        Returns True (always active) when schedule config is absent or empty.
    """
    if schedule_config is None:
        schedule_config = _get_schedule_config()

    if not schedule_config:
        return True  # No schedule = 24/7 active

    # Resolve current time in the configured timezone
    now = _get_now(schedule_config.get("timezone"))

    # Check active_days (1=Monday .. 7=Sunday, matching ISO weekday)
    active_days = schedule_config.get("active_days")
    if active_days and isinstance(active_days, (list, tuple)):
        iso_weekday = now.isoweekday()  # 1=Mon, 7=Sun
        if iso_weekday not in active_days:
            logger.debug(
                f"Schedule: day {iso_weekday} not in active_days {active_days}"
            )
            return False

    # Check active_hours
    active_hours = schedule_config.get("active_hours")
    if active_hours:
        time_range = _parse_time_range(active_hours)
        if time_range is None:
            # Invalid format — treat as always active (fail-open)
            logger.warning(
                f"Schedule: invalid active_hours '{active_hours}', treating as always active"
            )
            return True

        start, end = time_range
        current_time = now.time().replace(microsecond=0)

        if start <= end:
            # Normal range: e.g. 09:00-23:00
            if not (start <= current_time <= end):
                logger.debug(
                    f"Schedule: {current_time} not in {start}-{end}"
                )
                return False
        else:
            # Overnight range: e.g. 22:00-06:00
            if not (current_time >= start or current_time <= end):
                logger.debug(
                    f"Schedule: {current_time} not in overnight {start}-{end}"
                )
                return False

    return True


def _get_now(timezone_str: Optional[str] = None) -> datetime:
    """Get current datetime, optionally in the specified timezone.

    Args:
        timezone_str: IANA timezone name (e.g. 'Asia/Seoul'). If None,
            uses system local time.

    Returns:
        Current datetime (timezone-aware if possible).
    """
    if timezone_str:
        try:
            # Python 3.9+ has zoneinfo in stdlib
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone_str)
            return datetime.now(tz)
        except ImportError:
            logger.debug("zoneinfo not available, trying dateutil")
            try:
                from dateutil import tz as dateutil_tz
                tz = dateutil_tz.gettz(timezone_str)
                if tz:
                    return datetime.now(tz)
            except ImportError:
                pass
            logger.warning(
                f"Cannot resolve timezone '{timezone_str}', using system local"
            )

    return datetime.now()
