"""Validate and ingest evidence-backed medicine interaction records."""
import argparse
import csv
import hashlib
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from dotenv import load_dotenv
from neo4j import GraphDatabase

from monitoring import log_event

load_dotenv(".env.local")
load_dotenv()
REQUIRED = ("medicine_a_id", "medicine_b_id", "severity", "evidence_source", "evidence_url", "evidence_summary", "jurisdiction", "reviewed_at", "license", "provenance_status")


def validate(row: dict[str, str], row_number: int) -> dict[str, str]:
    missing = [key for key in REQUIRED if not (row.get(key) or "").strip()]
    if missing:
        raise ValueError(f"row {row_number}: missing {', '.join(missing)}")
    a, b = row["medicine_a_id"].strip(), row["medicine_b_id"].strip()
    if a == b:
        raise ValueError(f"row {row_number}: interaction endpoints must differ")
    parsed = urlparse(row["evidence_url"].strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"row {row_number}: evidence_url must be an https URL")
    if row["provenance_status"].strip().lower() not in {"verified", "reviewed"}:
        raise ValueError(f"row {row_number}: provenance_status must be verified or reviewed")
    row["source_date"] = (row.get("source_date") or "unknown").strip()
    row["retrieved_at"] = (row.get("retrieved_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()).strip()
    for field in ("source_date", "retrieved_at", "reviewed_at"):
        value = (row.get(field) or "").strip()
        if value.lower() != "unknown" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T[^\s]+)?", value):
            raise ValueError(f"row {row_number}: {field} must be an ISO date or timestamp")
    pair = sorted((a, b))
    evidence_identity = "|".join([*pair, row["evidence_url"].strip(), row["evidence_source"].strip()])
    row["interaction_id"] = (row.get("interaction_id") or "").strip() or "ddi_" + hashlib.sha256(evidence_identity.encode()).hexdigest()[:24]
    row["source_id"] = (row.get("source_id") or "").strip() or "src_" + hashlib.sha256(row["evidence_url"].strip().encode()).hexdigest()[:24]
    return {key: (value or "").strip() or None for key, value in row.items()}


def ingest(path: str) -> int:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError(f"CSV must include: {', '.join(REQUIRED)}")
        rows = [validate(row, index) for index, row in enumerate(reader, start=2)]
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all((uri, user, password)):
        raise ValueError("Missing Neo4j credentials in .env.local or .env")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for row in rows:
                result = session.run("""
                    MATCH (a:Medicine {medicine_id:$medicine_a_id})
                    MATCH (b:Medicine {medicine_id:$medicine_b_id})
                    MERGE (s:DataSource {source_id:$source_id})
                    SET s.name=$evidence_source, s.url=$evidence_url,
                        s.source_date=$source_date, s.retrieved_at=$retrieved_at,
                        s.license=$license, s.jurisdiction=$jurisdiction,
                        s.content_type='medicine_interaction_evidence',
                        s.provenance_status=$provenance_status
                    MERGE (a)-[r:HAS_KNOWN_INTERACTION {interaction_id:$interaction_id}]-(b)
                    SET r.severity=$severity, r.evidence_source=$evidence_source,
                        r.source_id=$source_id,
                        r.evidence_url=$evidence_url, r.evidence_summary=$evidence_summary,
                        r.jurisdiction=$jurisdiction, r.source_date=$source_date,
                        r.retrieved_at=$retrieved_at, r.reviewed_at=$reviewed_at,
                        r.license=$license, r.provenance_status=$provenance_status
                    MERGE (a)-[:SUPPORTED_BY {interaction_id:$interaction_id}]->(s)
                    MERGE (b)-[:SUPPORTED_BY {interaction_id:$interaction_id}]->(s)
                    RETURN a.medicine_id AS a, b.medicine_id AS b
                """, **row)
                if not result.single():
                    raise ValueError(f"medicine ID not found: {row['medicine_a_id']} or {row['medicine_b_id']}")
        log_event("interaction_ingestion_completed", rows=len(rows), evidence_required=True)
        return len(rows)
    except Exception as exc:
        log_event("interaction_ingestion_failed", error_type=type(exc).__name__)
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="data/interactions_template.csv")
    args = parser.parse_args()
    print(f"Ingested {ingest(args.csv)} evidence-backed interactions")
