"""Audit and import the staged Ayurveda CSV into isolated Neo4j labels.

AyurvedaEntry nodes retain the original row verbatim. MENTIONS_HERB edges encode
only that an herb name appears in a row; they do not encode clinical efficacy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipeline import normalize_display, normalize_key
from monitoring import log_event

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data/ayurveda/ayurveda_dataset.csv"
DEFAULT_MANIFEST = ROOT / "data/ayurveda/source_manifest.json"
REPORT_DIR = ROOT / "reports/ayurveda"
REQUIRED_COLUMNS = ("ID", "Problem", "Symptoms", "Remedies", "Medicines", "Source")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_dataset(csv_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    raw = csv_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    for key in ("source_id", "source_name", "license", "content_type"):
        if not manifest.get(key):
            raise ValueError(f"Manifest is missing {key}")

    records = []
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))
        for row_number, raw_row in enumerate(reader, start=2):
            original = {str(k): (v or "") for k, v in raw_row.items() if k is not None}
            canonical = {k: normalize_display(v) for k, v in sorted(original.items())}
            row_digest = hashlib.sha256(json.dumps(
                [manifest["source_id"], canonical], ensure_ascii=False,
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            herbs = []
            seen_herbs = set()
            for herb in re.split(r"[,;]", original.get("Medicines", "")):
                herb = normalize_display(herb)
                key = normalize_key(herb)
                if herb and key and key not in seen_herbs:
                    herbs.append({"name": herb, "normalized_name": key})
                    seen_herbs.add(key)
            try:
                confidence = float(original.get("Confidence", ""))
            except (TypeError, ValueError):
                confidence = None
            try:
                authentication_score = float(original.get("Auth_Score", ""))
            except (TypeError, ValueError):
                authentication_score = None
            records.append({
                "record_id": "ayurveda_" + row_digest[:24],
                "source_row_id": "row_" + row_digest[:24],
                "source_row_number": row_number,
                "problem": normalize_display(original.get("Problem")),
                "normalized_problem": normalize_key(original.get("Problem")),
                "symptoms": normalize_display(original.get("Symptoms")),
                "normalized_symptoms": normalize_key(original.get("Symptoms")),
                "medicines_text": normalize_display(original.get("Medicines")),
                "normalized_medicines": normalize_key(original.get("Medicines")),
                "herbs": herbs,
                "source_reference": normalize_display(original.get("Source")),
                "confidence_value": confidence,
                "authentication_score": authentication_score,
                "source_record_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
                "source_id": manifest["source_id"],
                "source_sha256": digest,
                "jurisdiction": manifest.get("jurisdiction") or "unknown",
                "source_date": manifest.get("source_date") or "unknown",
                "retrieved_at": manifest.get("retrieved_at") or "unknown",
                "license": manifest["license"],
                "content_type": manifest["content_type"],
                "clinical_authority": manifest.get("clinical_authority", "not_assessed"),
                "provenance_status": "unverified_preview",
                "ingested_at": utc_now(),
            })
    return records, manifest, digest


def audit(records: list[dict[str, Any]], csv_path: Path, digest: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    problem_counts = Counter()
    herb_counts = Counter()
    source_counts = Counter()
    empty_counts = Counter()
    confidence_values = Counter()
    for record in records:
        original = json.loads(record["source_record_json"])
        content = {key: normalize_display(value) for key, value in original.items() if key != "ID"}
        content_digest = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        exact_groups[content_digest].append(record)
        problem_counts[record["normalized_problem"]] += 1
        source_counts[record["source_reference"]] += 1
        for herb in record["herbs"]:
            herb_counts[herb["normalized_name"]] += 1
        raw = json.loads(record["source_record_json"])
        for key, value in raw.items():
            if not normalize_display(value):
                empty_counts[key] += 1
        if record["confidence_value"] is not None:
            confidence_values[str(record["confidence_value"])] += 1
    duplicates = []
    for identity, group in exact_groups.items():
        if len(group) > 1:
            duplicates.append({"content_sha256": identity, "count": len(group), "row_numbers": [x["source_row_number"] for x in group]})
    repeated_problem_groups = sum(1 for count in problem_counts.values() if count > 1)
    field_values: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for key, value in json.loads(record["source_record_json"]).items():
            field_values[key].add(normalize_display(value))
    constant_nonempty_fields = {
        key: next(iter(values)) for key, values in sorted(field_values.items())
        if len(values) == 1 and next(iter(values))
    }
    report = {
        "generated_at": utc_now(), "source_path": str(csv_path.relative_to(ROOT)),
        "source_sha256": digest, "source_rows": len(records),
        "unique_record_ids": len({record["record_id"] for record in records}), "exact_duplicate_content_groups_ignoring_id": len(duplicates),
        "unique_normalized_problem_labels": len(problem_counts),
        "problem_labels_repeated_in_multiple_rows": repeated_problem_groups,
        "unique_herb_mentions": len(herb_counts), "distinct_source_reference_strings": len(source_counts),
        "empty_values_by_column": dict(sorted(empty_counts.items())),
        "confidence_value_distribution": dict(sorted(confidence_values.items())),
        "constant_nonempty_fields_across_all_rows": constant_nonempty_fields,
        "most_common_problem_labels": problem_counts.most_common(20),
        "most_common_herb_mentions": herb_counts.most_common(30),
        "most_common_source_reference_strings": source_counts.most_common(20),
        "provenance_status": "unverified_preview",
        "notes": [
            "Source and license are undocumented in this repository; verify before redistribution.",
            "Repeated rows and labels are retained; row-level claims have not been clinically reviewed.",
            "Confidence and authentication score columns are copied as supplied, not interpreted as evidence quality.",
        ],
    }
    return report, duplicates


def write_audit(report: dict[str, Any], duplicates: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (REPORT_DIR / "duplicate_review.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("content_sha256", "count", "row_numbers"))
        writer.writeheader()
        for item in duplicates:
            writer.writerow({**item, "row_numbers": ",".join(map(str, item["row_numbers"]))})


def ingest(records: list[dict[str, Any]], manifest: dict[str, Any], digest: str, batch_size: int = 500) -> None:
    load_dotenv(ROOT / ".env.local")
    load_dotenv()
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all((uri, user, password)):
        raise ValueError("Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in .env.local")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    now = utc_now()
    source = {
        "source_id": manifest["source_id"], "name": manifest["source_name"],
        "url": manifest.get("source_url"), "source_date": manifest.get("source_date") or "unknown",
        "retrieved_at": manifest.get("retrieved_at") or "unknown", "license": manifest["license"],
        "sha256": digest, "jurisdiction": manifest.get("jurisdiction") or "unknown",
        "content_type": manifest["content_type"], "clinical_authority": manifest.get("clinical_authority", "not_assessed"),
        "provenance_status": "unverified_preview", "ingested_at": now,
    }
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            session.run("CREATE CONSTRAINT ayurveda_entry_id IF NOT EXISTS FOR (e:AyurvedaEntry) REQUIRE e.record_id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT ayurveda_source_id IF NOT EXISTS FOR (s:DataSource) REQUIRE s.source_id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT ayurvedic_herb_name IF NOT EXISTS FOR (h:AyurvedicHerb) REQUIRE h.normalized_name IS UNIQUE").consume()
            session.run("""
                MATCH (s:DataSource {source_id:$source_id})
                WHERE s.sha256 <> $sha256
                WITH s
                MATCH (e:AyurvedaEntry)-[:DERIVED_FROM {source_id:$source_id}]->(s)
                DETACH DELETE e
            """, {"source_id": source["source_id"], "sha256": digest}).consume()
            session.run("""
                MERGE (s:DataSource {source_id:$source_id}) SET s += $source
            """, {"source_id": source["source_id"], "source": source}).consume()
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                session.run("""
                    UNWIND $rows AS row
                    MERGE (e:AyurvedaEntry {record_id:row.entry.record_id})
                    SET e += row.entry
                    WITH e, row
                    MATCH (s:DataSource {source_id:row.entry.source_id})
                    MERGE (e)-[d:DERIVED_FROM {source_id:row.entry.source_id}]->(s)
                    SET d.source_row_id=row.entry.source_row_id,
                        d.source_row_number=row.entry.source_row_number,
                        d.source_sha256=row.entry.source_sha256
                    WITH e, row
                    UNWIND row.herbs AS herb
                    MERGE (h:AyurvedicHerb {normalized_name:herb.normalized_name})
                    ON CREATE SET h.name=herb.name, h.provenance_status='unverified_preview'
                    MERGE (e)-[r:MENTIONS_HERB]->(h)
                    SET r.source_id=row.entry.source_id,
                        r.source_row_id=row.entry.source_row_id,
                        r.provenance_status='unverified_preview'
                """, {"rows": [{"entry": {k: v for k, v in record.items() if k != "herbs"}, "herbs": record["herbs"]} for record in batch]}).consume()
        log_event("ayurveda_ingestion_completed", source_id=source["source_id"], source_sha256=digest, source_rows=len(records))
    except Exception as exc:
        log_event("ayurveda_ingestion_failed", source_id=source["source_id"], source_sha256=digest, error_type=type(exc).__name__)
        raise
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and optionally import Ayurveda CSV records into Neo4j")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ingest", action="store_true", help="Write isolated AyurvedaEntry/AyurvedicHerb records to Neo4j")
    parser.add_argument("--limit", type=int, help="Import only the first N rows, for a small staging run")
    args = parser.parse_args()
    records, manifest, digest = read_dataset(args.csv, args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        records = records[:args.limit]
    report, duplicates = audit(records, args.csv, digest)
    write_audit(report, duplicates)
    print(json.dumps({key: report[key] for key in (
        "source_rows", "unique_record_ids", "exact_duplicate_content_groups_ignoring_id",
        "unique_normalized_problem_labels", "unique_herb_mentions", "source_sha256",
    )}, indent=2))
    print(f"Audit written to {REPORT_DIR}")
    if args.ingest:
        ingest(records, manifest, digest)
        print(f"Imported {len(records)} unverified Ayurveda preview rows to Neo4j.")
    else:
        print("Audit only. Pass --ingest to write these records to Neo4j.")


if __name__ == "__main__":
    main()
