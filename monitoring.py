"""Small privacy-conscious local event logger and source/citation checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()


def source_freshness(retrieved_at: str | None, max_age_days: int = 180) -> dict[str, Any]:
    if not retrieved_at:
        return {"status": "unknown", "age_days": None}
    try:
        parsed = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return {"status": "invalid_date", "age_days": None}
    age_days = max(0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).days)
    return {"status": "stale" if age_days > max_age_days else "fresh", "age_days": age_days}


def audit_citations(answer: str, source_ids: Iterable[str]) -> dict[str, Any]:
    allowed = {str(source_id) for source_id in source_ids}
    cited = set(re.findall(r"\[(S\d+)\]", answer or ""))
    return {
        "citation_ids": sorted(cited),
        "unknown_citation_ids": sorted(cited - allowed),
        "has_citation": bool(cited),
        "all_citations_resolve": bool(cited) and cited.issubset(allowed),
        "source_count": len(allowed),
    }


def log_event(event_type: str, **fields: Any) -> None:
    """Append one JSON event without storing user query text or model output."""
    destination = Path(os.getenv("MEDGRAPH_LOG_PATH", "runtime/medgraph_events.jsonl"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": utc_now(), "event_type": event_type, **fields}
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    descriptor = os.open(destination, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)
