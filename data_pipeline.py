"""Deterministic normalization, identity, and source-data quality reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).parent / "data" / "source_manifest.json"
REQUIRED_COLUMNS = (
    "Medicine Name",
    "Composition",
    "Uses",
    "Side_effects",
    "Image URL",
    "Manufacturer",
)
REVIEW_COLUMNS = (
    "Excellent Review %",
    "Average Review %",
    "Poor Review %",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_display(value: Any) -> str:
    """Normalize Unicode and whitespace while preserving punctuation and case."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: Any) -> str:
    """Produce a case- and punctuation-insensitive key for matching/review."""
    text = unicodedata.normalize("NFKD", normalize_display(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_ingredients(composition: Any) -> list[str]:
    """Extract ingredient names while retaining the unmodified composition separately."""
    result: list[str] = []
    for part in normalize_display(composition).split("+"):
        ingredient = re.sub(r"\([^)]*\)", " ", part)
        ingredient = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|iu|units?|%)(?:\s*/\s*[a-z]+)?\b", " ", ingredient, flags=re.I)
        ingredient = normalize_display(ingredient).strip(" -,:;/")
        key = normalize_key(ingredient)
        if ingredient and key and all(normalize_key(old) != key for old in result):
            result.append(ingredient)
    return result


def stable_medicine_id(name: Any, composition: Any, manufacturer: Any) -> str:
    """Stable identity for a catalog product/formulation, independent of row order."""
    identity = [normalize_key(name), normalize_key(composition), normalize_key(manufacturer)]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"med_{digest[:24]}"


def source_row_id(source_id: str, row: dict[str, Any]) -> str:
    canonical = {key: normalize_display(value) for key, value in sorted(row.items())}
    payload = json.dumps([source_id, canonical], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "row_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source_file:
        manifest = json.load(source_file)
    required = ("source_id", "source_name", "license", "content_type")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"Source manifest is missing required values: {', '.join(missing)}")
    return manifest


def read_source_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            rows.append({"_source_row_number": row_number, **row})
        return rows


def _review_numbers(row: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    values: list[int | None] = []
    for column in REVIEW_COLUMNS:
        raw = normalize_display(row.get(column))
        try:
            values.append(int(raw) if raw else None)
        except ValueError:
            values.append(None)
    return tuple(values)  # type: ignore[return-value]


def prepare_records(
    csv_path: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    ingested_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Return deduplicated product records, a quality report, and duplicate review rows.

    Exact product duplicates are collapsed deterministically while their source row IDs
    and row numbers are retained. Same-name products with different composition or
    manufacturer remain separate products and are emitted for manual review.
    """
    csv_path = Path(csv_path)
    raw_bytes = csv_path.read_bytes()
    raw_rows = read_source_rows(csv_path)
    manifest = load_manifest(manifest_path)
    loaded_at = ingested_at or utc_now()

    product_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    name_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completeness: Counter[str] = Counter()
    review_sum_counts: Counter[str] = Counter()
    invalid_review_rows: list[int] = []
    all_source_row_ids: set[str] = set()

    for row in raw_rows:
        display = {column: normalize_display(row.get(column)) for column in row if not column.startswith("_")}
        original = {column: (row.get(column) or "") for column in row if not column.startswith("_")}
        name = display.get("Medicine Name", "")
        composition = display.get("Composition", "")
        manufacturer = display.get("Manufacturer", "") or "Unknown"
        medicine_id = stable_medicine_id(name, composition, manufacturer)
        row_id = source_row_id(manifest["source_id"], display)
        prepared = {
            "medicine_id": medicine_id,
            "name": name,
            "normalized_name": normalize_key(name),
            "composition": composition,
            "normalized_composition": normalize_key(composition),
            "manufacturer": manufacturer,
            "normalized_manufacturer": normalize_key(manufacturer),
            "uses_text": display.get("Uses", ""),
            "side_effects_text": display.get("Side_effects", ""),
            "image_url": display.get("Image URL", ""),
            "ingredients": parse_ingredients(composition),
            "normalized_ingredients": [normalize_key(item) for item in parse_ingredients(composition)],
            "excellent_review_pct": _review_numbers(row)[0],
            "average_review_pct": _review_numbers(row)[1],
            "poor_review_pct": _review_numbers(row)[2],
            "source_record_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
            "source_row_id": row_id,
            "source_row_number": row["_source_row_number"],
        }
        product_groups[medicine_id].append(prepared)
        name_groups[prepared["normalized_name"]].append(prepared)
        all_source_row_ids.add(row_id)
        for column in REQUIRED_COLUMNS + REVIEW_COLUMNS:
            if normalize_display(row.get(column)):
                completeness[column] += 1
        reviews = prepared["excellent_review_pct"], prepared["average_review_pct"], prepared["poor_review_pct"]
        if all(value is not None for value in reviews):
            review_sum_counts[str(sum(value for value in reviews if value is not None))] += 1
        else:
            invalid_review_rows.append(row["_source_row_number"])

    products: list[dict[str, Any]] = []
    for medicine_id, variants in sorted(product_groups.items()):
        representative = dict(variants[0])
        unique_source_variants = {row["source_row_id"]: row["source_record_json"] for row in variants}
        representative["source_row_ids"] = sorted({row["source_row_id"] for row in variants})
        representative["source_row_numbers"] = sorted({row["source_row_number"] for row in variants})
        representative["source_row_count"] = len(variants)
        representative["source_variants_json"] = json.dumps(
            [json.loads(value) for _, value in sorted(unique_source_variants.items())],
            ensure_ascii=False,
            sort_keys=True,
        )
        conflicting_fields = []
        for field in ("uses_text", "side_effects_text", "image_url", "excellent_review_pct", "average_review_pct", "poor_review_pct"):
            if len({row[field] for row in variants}) > 1:
                conflicting_fields.append(field)
        representative["source_conflict_fields"] = conflicting_fields
        representative.update({
            "source_id": manifest["source_id"],
            "source_name": manifest["source_name"],
            "source_url": manifest.get("source_url"),
            "source_date": manifest.get("source_date"),
            "source_retrieved_at": manifest.get("retrieved_at"),
            "source_license": manifest.get("license"),
            "jurisdiction": manifest.get("jurisdiction"),
            "content_type": manifest.get("content_type"),
            "clinical_authority": manifest.get("clinical_authority", "not_assessed"),
            "provenance_status": manifest.get("provenance_status", "incomplete"),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "ingested_at": loaded_at,
        })
        products.append(representative)

    duplicate_review: list[dict[str, Any]] = []
    repeated_name_groups = 0
    variant_name_groups = 0
    repeated_product_groups = 0
    conflicting_product_groups = 0
    for medicine_id, variants in sorted(product_groups.items()):
        unique_payloads = {row["source_row_id"] for row in variants}
        if len(variants) > 1:
            repeated_product_groups += 1
            field_conflicts = [
                field for field in ("uses_text", "side_effects_text", "image_url", "excellent_review_pct", "average_review_pct", "poor_review_pct")
                if len({row[field] for row in variants}) > 1
            ]
            if field_conflicts:
                conflicting_product_groups += 1
            duplicate_review.append({
                "normalized_name": variants[0]["normalized_name"],
                "medicine_id": medicine_id,
                "name": variants[0]["name"],
                "composition": variants[0]["composition"],
                "manufacturer": variants[0]["manufacturer"],
                "source_row_number": ",".join(str(row["source_row_number"]) for row in variants),
                "reason": (
                    "same product identity has conflicting source values: " + ", ".join(field_conflicts)
                    if field_conflicts else "duplicate product identity; repeated source rows are retained in provenance"
                ),
            })
    for normalized_name, variants in sorted(name_groups.items()):
        ids = sorted({row["medicine_id"] for row in variants})
        if len(variants) > 1:
            repeated_name_groups += 1
        if len(ids) > 1:
            variant_name_groups += 1
            for row in variants:
                duplicate_review.append({
                    "normalized_name": normalized_name,
                    "medicine_id": row["medicine_id"],
                    "name": row["name"],
                    "composition": row["composition"],
                    "manufacturer": row["manufacturer"],
                    "source_row_number": row["source_row_number"],
                    "reason": "same normalized name has multiple product identities; review before any merge",
                })

    report = {
        "generated_at": loaded_at,
        "source_id": manifest["source_id"],
        "source_name": manifest["source_name"],
        "source_path": str(csv_path),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_url": manifest.get("source_url"),
        "source_date": manifest.get("source_date"),
        "source_retrieved_at": manifest.get("retrieved_at"),
        "license": manifest.get("license"),
        "jurisdiction": manifest.get("jurisdiction"),
        "provenance_status": manifest.get("provenance_status", "incomplete"),
        "source_rows": len(raw_rows),
        "unique_product_ids": len(products),
        "exact_duplicate_rows": len(raw_rows) - len(all_source_row_ids),
        "same_product_identity_extra_rows": len(raw_rows) - len(products),
        "repeated_product_identity_groups": repeated_product_groups,
        "conflicting_product_identity_groups": conflicting_product_groups,
        "repeated_normalized_name_groups": repeated_name_groups,
        "multi_identity_name_groups_for_review": variant_name_groups,
        "completeness_counts": dict(completeness),
        "review_percentage_sum_counts": dict(review_sum_counts),
        "invalid_or_missing_review_rows": invalid_review_rows,
        "safety_notice": "Catalog metadata only; source authority and jurisdiction are not verified.",
    }
    return products, report, duplicate_review


def write_audit(csv_path: str | Path, output_dir: str | Path, manifest_path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    products, report, duplicate_review = prepare_records(csv_path, manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["duplicate_review_rows"] = len(duplicate_review)
    report["prepared_product_count"] = len(products)
    with open(output_dir / "data_quality_report.json", "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2, ensure_ascii=False)
    if duplicate_review:
        with open(output_dir / "duplicate_review.csv", "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(duplicate_review[0]))
            writer.writeheader()
            writer.writerows(duplicate_review)
    else:
        with open(output_dir / "duplicate_review.csv", "w", newline="", encoding="utf-8") as target:
            target.write("normalized_name,medicine_id,name,composition,manufacturer,source_row_number,reason\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and audit MEDGraphy medicine source data")
    parser.add_argument("--csv", default="data/Medicine_Details.csv")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="reports/data_quality")
    args = parser.parse_args()
    report = write_audit(args.csv, args.output_dir, args.manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
