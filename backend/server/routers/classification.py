"""
Classification domain management API — admin only.

Provides endpoints to list, inspect, and activate domain presets
for the domain-aware classification system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.server.deps import require_admin
from backend.vision.domain_loader import list_available_domains, load_domain
from backend.utils.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/classification", tags=["classification"])

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ── Response Models ──────────────────────────────────────────

class DomainSummary(BaseModel):
    id: str
    name: str
    name_ko: str
    description: str
    image_types_count: int
    image_types: List[str]


class DomainDetail(BaseModel):
    id: str
    name: str
    name_ko: str
    description: str
    image_types: List[str]
    type_hints: Dict[str, Dict[str, List[str]]]
    type_instructions: Dict[str, str]
    common_hints: Dict[str, List[str]]


class ActiveDomainResponse(BaseModel):
    active_domain: Optional[str] = None
    domain: Optional[DomainDetail] = None


class SetActiveDomainRequest(BaseModel):
    domain_id: str


class CreateDomainRequest(BaseModel):
    domain_id: str
    yaml_content: str


# ── Endpoints ────────────────────────────────────────────────

@router.get("/domains", response_model=List[DomainSummary])
def get_domains(admin: dict = Depends(require_admin)):
    """List all available domain presets."""
    return list_available_domains()


@router.get("/domains/{domain_id}", response_model=DomainDetail)
def get_domain_detail(domain_id: str, admin: dict = Depends(require_admin)):
    """Get full details of a specific domain preset."""
    try:
        profile = load_domain(domain_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return DomainDetail(
        id=profile.id,
        name=profile.name,
        name_ko=profile.name_ko,
        description=profile.description,
        image_types=profile.image_types,
        type_hints=profile.type_hints,
        type_instructions=profile.type_instructions,
        common_hints=profile.common_hints,
    )


@router.get("/active", response_model=ActiveDomainResponse)
def get_active(admin: dict = Depends(require_admin)):
    """Get the currently active domain."""
    cfg = get_config()
    active_id = cfg.get("classification.active_domain")

    domain = None
    if active_id:
        try:
            profile = load_domain(active_id)
            domain = DomainDetail(
                id=profile.id,
                name=profile.name,
                name_ko=profile.name_ko,
                description=profile.description,
                image_types=profile.image_types,
                type_hints=profile.type_hints,
                type_instructions=profile.type_instructions,
                common_hints=profile.common_hints,
            )
        except Exception:
            logger.warning(f"Active domain '{active_id}' failed to load")

    return ActiveDomainResponse(active_domain=active_id, domain=domain)


@router.put("/active")
def set_active(
    req: SetActiveDomainRequest,
    admin: dict = Depends(require_admin),
):
    """Set the active classification domain. Updates config.yaml."""
    # Validate domain exists
    domains = list_available_domains()
    valid_ids = [d["id"] for d in domains]
    if req.domain_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain: {req.domain_id}. Available: {valid_ids}",
        )

    # Update config.yaml
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        data.setdefault("classification", {})["active_domain"] = req.domain_id

        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Reset config singleton so next read picks up the change
        import backend.utils.config as config_module
        config_module._instance = None

        logger.info(f"Active domain changed to '{req.domain_id}' by admin '{admin['username']}'")
        return {"success": True, "active_domain": req.domain_id}

    except Exception as e:
        logger.error(f"Failed to update active domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@router.post("/domains", status_code=201)
def create_domain(
    req: CreateDomainRequest,
    admin: dict = Depends(require_admin),
):
    """Create a new domain profile from raw YAML content."""
    import re

    # 1. Validate domain_id format
    if not re.match(r'^[a-z][a-z0-9_]*$', req.domain_id):
        raise HTTPException(400, "Invalid domain ID: must be lowercase snake_case")

    # 2. Check if already exists
    domains_dir = _PROJECT_ROOT / "backend" / "vision" / "domains"
    target_path = domains_dir / f"{req.domain_id}.yaml"
    if target_path.exists():
        raise HTTPException(409, f"Domain '{req.domain_id}' already exists")

    # 3. Parse YAML
    try:
        parsed = yaml.safe_load(req.yaml_content)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"YAML parse error: {str(e)}")

    # 4. Structural validation
    if not isinstance(parsed, dict):
        raise HTTPException(400, "YAML must be a valid object")

    domain_meta = parsed.get("domain")
    if not domain_meta or not domain_meta.get("id"):
        raise HTTPException(400, "Missing required field: domain.id")

    if domain_meta["id"] != req.domain_id:
        raise HTTPException(
            400,
            f"domain.id mismatch: expected '{req.domain_id}', got '{domain_meta['id']}'",
        )

    image_types = parsed.get("image_types")
    if not isinstance(image_types, list) or len(image_types) == 0:
        raise HTTPException(400, "image_types must be a non-empty array")

    if not isinstance(parsed.get("type_hints", {}), dict):
        raise HTTPException(400, "type_hints must be an object")

    # 5. Write file
    try:
        domains_dir.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(
                parsed, f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

        logger.info(f"Domain '{req.domain_id}' created by admin '{admin['username']}'")
        return {"success": True, "domain_id": req.domain_id}

    except Exception as e:
        logger.error(f"Failed to save domain: {e}")
        raise HTTPException(500, f"Failed to save domain: {str(e)}")
