"""Phase D: persist user 'irrelevant' labels for search results.

Recorded labels feed soft demotion at search time so a user-marked
file_id drops in subsequent searches for the same query.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_current_user, get_db_safe

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search-feedback"])


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    file_id: int
    label: Literal["irrelevant"] = "irrelevant"


@router.post("/search/feedback")
def submit_feedback(
    req: FeedbackRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    db.conn.execute(
        """INSERT INTO search_feedback (query, file_id, label, user_id)
           VALUES (?, ?, ?, ?)""",
        (req.query, req.file_id, req.label, user["id"]),
    )
    db.conn.commit()
    logger.info(
        "search_feedback: user=%s query=%r file_id=%s label=%s",
        user.get("username"), req.query[:80], req.file_id, req.label,
    )
    return {"success": True}
