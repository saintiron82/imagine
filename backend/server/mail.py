"""
Pluggable email sender for member invites.

If SMTP is configured (server.smtp.{host,port,username,password,from} in
config.yaml or IMAGINE_SMTP_* env), invites are emailed. Otherwise sending is a
no-op and the caller surfaces the invite link in the UI for manual sharing —
so the feature degrades gracefully without external SMTP credentials.
"""
import logging
import os
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def _smtp_config() -> dict:
    try:
        from backend.server.config import get_server_config
        cfg = dict(get_server_config().get("smtp", {}) or {})
    except Exception:
        cfg = {}
    for key, env in (
        ("host", "IMAGINE_SMTP_HOST"), ("port", "IMAGINE_SMTP_PORT"),
        ("username", "IMAGINE_SMTP_USER"), ("password", "IMAGINE_SMTP_PASSWORD"),
        ("from", "IMAGINE_SMTP_FROM"),
    ):
        v = os.getenv(env)
        if v:
            cfg[key] = v
    return cfg


def smtp_configured() -> bool:
    cfg = _smtp_config()
    return bool(cfg.get("host"))


def send_invite_email(to_email: str, link: str, group_name: str = "Imagine") -> bool:
    """Send an invite email. Returns True if sent, False if SMTP not configured/failed."""
    cfg = _smtp_config()
    if not cfg.get("host"):
        logger.info(f"[mail] SMTP not configured; invite link for {to_email} returned to UI")
        return False
    try:
        body = (
            f"{group_name} 에 초대되었습니다.\n\n"
            f"아래 링크에서 수락하세요 (이 이메일로만 수락 가능):\n{link}\n"
        )
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = f"[{group_name}] 초대"
        msg["From"] = cfg.get("from") or cfg.get("username") or "no-reply@imagine.app"
        msg["To"] = to_email
        port = int(cfg.get("port", 587))
        with smtplib.SMTP(cfg["host"], port, timeout=10) as s:
            s.starttls()
            if cfg.get("username"):
                s.login(cfg["username"], cfg.get("password", ""))
            s.send_message(msg)
        logger.info(f"[mail] invite sent to {to_email}")
        return True
    except Exception as e:
        logger.warning(f"[mail] invite send failed for {to_email}: {e}")
        return False
