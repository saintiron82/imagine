"""Single source for derivation model_version strings (CAS design §3.2).

The version must capture everything that changes a phase's OUTPUT:
- mc: VLM model id + prompt contract version + resolved domain profile
- vv: visual embedding model id + dimensions
- mv: text embedding model id + dimensions

The domain part is a content fingerprint of the *resolved* profile
(including user overrides from config.yaml), so editing a domain YAML
changes the version automatically — no human discipline required.

Known caveat (M4 refinement): platform backends (mlx 4-bit vs ollama vs
transformers) of the same logical VLM produce slightly different output.
The cluster already mixes them in `files`, so the canonical model id
matches existing semantics.
"""

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Bump when the VLM prompt contract changes. Mirrors the
# `prompt_version` literal reported in _vlm_provenance by the adapters.
MC_PROMPT_VERSION = "spatial_v2"

PHASES = ("mc", "vv", "mv")


def _domain_fingerprint() -> str:
    """Name + content hash of the resolved active domain profile."""
    try:
        from backend.vision.domain_loader import get_active_domain
        domain = get_active_domain()
        if domain is None:
            return "nodomain"
        try:
            import dataclasses
            payload = dataclasses.asdict(domain)
        except (TypeError, ValueError):
            payload = getattr(domain, "__dict__", str(domain))
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:8]
        name = getattr(domain, "name", None) or getattr(domain, "domain_id", "domain")
        return f"{name}-{digest}"
    except Exception as e:
        logger.debug(f"domain fingerprint unavailable: {e}")
        return "nodomain"


def get_model_version(phase: str) -> str:
    """Version string for a phase's derivations. Deterministic per config."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")

    from backend.utils.tier_config import get_active_tier
    _, cfg = get_active_tier()

    if phase == "mc":
        model = (cfg.get("vlm") or {}).get("model", "unknown")
        return f"{model}/{MC_PROMPT_VERSION}/{_domain_fingerprint()}"
    if phase == "vv":
        v = cfg.get("visual") or {}
        return f"{v.get('model', 'unknown')}/d{v.get('dimensions', 0)}"
    # mv
    t = cfg.get("text_embed") or {}
    return f"{t.get('model', 'unknown')}/d{t.get('dimensions', 0)}"
