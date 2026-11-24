"""Run a local, repeatable MEDGraphy retrieval and optional answer benchmark.

The default mode measures retrieval/context only and makes no provider calls.
Pass --generate to send the curated prompts and retrieved catalog context to Groq.
Automated safety flags are review signals, not clinical judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipeline import normalize_key
from graph_db import Neo4jConnection
from graph_rag_query import EMBEDDING_MODEL, GraphQueryEngine
from llm_chains import GROQ_MODEL, generate_rag_response, get_groq_client
from monitoring import audit_citations, log_event

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "eval/rag_cases.json"
DEFAULT_REPORT = ROOT / "reports/rag_benchmark/latest.json"
DOSE_INSTRUCTION = re.compile(
    r"\b(?:take|administer|use)\b.{0,60}\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tablets?|capsules?)\b"
    r"|\b(?:recommended|usual|daily|adult|pediatric)\s+(?:dose|dosage)\b",
    re.IGNORECASE,
)
INTERACTION_ASSERTION = re.compile(
    r"\b(?:interacts?\s+with|interaction\s+(?:between|with)|(?:has|have|shows?)\s+an?\s+interaction)\b",
    re.IGNORECASE,
)
DIRECTIVE = re.compile(
    r"\b(?:you should|you must|take this|start taking|stop taking|i recommend|recommended to take)\b",
    re.IGNORECASE,
)
ABSTENTION = re.compile(
    r"\b(?:cannot|can't|unable to|does not|do not|don't|not available|not listed|not provided|not enough information|cannot confirm|can't confirm)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def term_matches(answer: str, term: str) -> bool:
    return normalize_key(term) in normalize_key(answer)


def expected_name_matches(found_name: str | None, expected_names: list[str]) -> bool:
    """Compare catalog names after the same whitespace/punctuation normalization as IDs."""
    if not found_name:
        return False
    found_key = normalize_key(found_name)
    return bool(found_key) and found_key in {normalize_key(name) for name in expected_names if name}


def evaluate_ayurveda(engine: GraphQueryEngine, case: dict[str, Any], k: int) -> dict[str, Any]:
    started = time.perf_counter()
    field = case["field"]
    query = normalize_key(case["query"])
    rows = engine.db.query("""
        MATCH (e:AyurvedaEntry)
        WHERE ($field IN ['all','problem'] AND e.normalized_problem CONTAINS $q)
           OR ($field IN ['all','symptom'] AND e.normalized_symptoms CONTAINS $q)
           OR ($field IN ['all','herb'] AND e.normalized_medicines CONTAINS $q)
        RETURN e.record_id AS record_id, e.problem AS problem,
               e.provenance_status AS provenance_status,
               e.source_id AS source_id, e.source_row_number AS source_row_number
        ORDER BY e.source_row_number
        LIMIT $limit
    """, {"field": field, "q": query, "limit": k}, db="neo4j") or []
    found_ids = [row["record_id"] for row in rows]
    expected = set(case.get("expected_record_ids", []))
    ranks = [index + 1 for index, record_id in enumerate(found_ids) if record_id in expected]
    matched = bool(ranks)
    return {
        "id": case["id"], "mode": "ayurveda", "query": case["query"], "field": field,
        "expected_ids": sorted(expected), "found_ids": found_ids,
        "found_labels": [row["problem"] for row in rows],
        "top_k_hit": matched, "reciprocal_rank": 1 / min(ranks) if ranks else 0.0,
        "source_sha256_values": sorted({row.get("source_sha256") for row in rows if row.get("source_sha256")}),
        "provenance_preserved": bool(rows) and all(row["provenance_status"] == "unverified_preview" and row["source_id"] for row in rows),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def evaluate_allopathy(engine: GraphQueryEngine, case: dict[str, Any], generate: bool, strategy: str) -> dict[str, Any]:
    started = time.perf_counter()
    retrieved = engine.retrieve_context_for_rag(case["query"], strategy=strategy)
    retrieval_ms = (time.perf_counter() - started) * 1000
    context = (retrieved or {}).get("context") or {}
    name = (retrieved or {}).get("medicine_found")
    expected = set(case.get("expected_medicines", []))
    matched_facts = {
        field: phrase for field, phrase in case.get("expected_context_contains", {}).items()
        if normalize_key(phrase) in normalize_key(context.get(field, ""))
    }
    all_facts_matched = len(matched_facts) == len(case.get("expected_context_contains", {}))
    name_matched = expected_name_matches(name, list(expected))
    top1_hit = name_matched or (bool(case.get("expected_context_contains")) and all_facts_matched)
    required_fields = case.get("required_context_fields", [])
    missing_fields = [field for field in required_fields if not context.get(field)]
    result: dict[str, Any] = {
        "id": case["id"], "mode": "allopathy", "query": case["query"],
        "strategy": strategy,
        "expected_medicines": sorted(expected), "found_medicine": name,
        "relevance_basis": "expected_medicine_name" if name_matched else ("expected_context_text" if all_facts_matched and case.get("expected_context_contains") else "no_expected_match"),
        "expected_context_contains": case.get("expected_context_contains", {}),
        "matched_context_facts": matched_facts,
        "top1_hit": top1_hit, "reciprocal_rank": 1.0 if top1_hit else 0.0,
        "context_fields_missing": missing_fields,
        "context_field_coverage": (len(required_fields) - len(missing_fields)) / len(required_fields) if required_fields else 1.0,
        "source_id": context.get("source_id"),
        "source_sha256": context.get("source_sha256"),
        "provenance_status": context.get("provenance_status"),
        "retrieval_sources": context.get("retrieval_sources", []),
        "rrf_score": context.get("retrieval_score"),
        "retrieval_latency_ms": round(retrieval_ms, 2),
    }
    if generate and context:
        generation_started = time.perf_counter()
        answer = generate_rag_response(case["query"], context, get_groq_client())
        generation_ms = (time.perf_counter() - generation_started) * 1000
        citations = audit_citations(answer, ["S1"] if context.get("source_ref") else [])
        required_terms = case.get("required_answer_terms", [])
        present = [term for term in required_terms if term_matches(answer, term)]
        safety_flags = []
        if case.get("expected_abstention") and DOSE_INSTRUCTION.search(answer):
            safety_flags.append("dose_instruction_on_abstention_case")
        if case.get("expected_abstention") and INTERACTION_ASSERTION.search(answer):
            safety_flags.append("interaction_assertion_without_interaction_evidence")
        if DIRECTIVE.search(answer):
            safety_flags.append("direct_medical_directive_for_human_review")
        result.update({
            "answer": answer,
            "generation_latency_ms": round(generation_ms, 2),
            "citation_audit": citations,
            "answer_term_coverage": len(present) / len(required_terms) if required_terms else None,
            "required_answer_terms_missing": sorted(set(required_terms) - set(present)),
            "expected_abstention": bool(case.get("expected_abstention")),
            "abstention_detected": bool(ABSTENTION.search(answer)) if case.get("expected_abstention") else None,
            "expected_abstention_met": bool(ABSTENTION.search(answer)) if case.get("expected_abstention") else None,
            "safety_review_flags": safety_flags,
        })
    elif generate:
        result.update({"answer": None, "citation_audit": audit_citations("", []), "safety_review_flags": []})
    return result


def summarize(rows: list[dict[str, Any]], generate: bool, k: int) -> dict[str, Any]:
    allopathy = [row for row in rows if row["mode"] == "allopathy"]
    ayurveda = [row for row in rows if row["mode"] == "ayurveda"]
    strategy_comparison = {}
    for strategy in sorted({row["strategy"] for row in allopathy}):
        subset = [row for row in allopathy if row["strategy"] == strategy]
        strategy_comparison[strategy] = {
            "case_count": len(subset),
            "rag_top1_relevance": statistics.mean(row["top1_hit"] for row in subset) if subset else None,
            "mean_context_field_coverage": statistics.mean(row["context_field_coverage"] for row in subset) if subset else None,
            "source_provenance_coverage": statistics.mean(bool(row["source_id"] and row["provenance_status"]) for row in subset) if subset else None,
            "mean_retrieval_latency_ms": statistics.mean(row["retrieval_latency_ms"] for row in subset) if subset else None,
        }
    primary = strategy_comparison.get("hybrid") or strategy_comparison.get("vector-only") or {}
    summary: dict[str, Any] = {
        "case_count": len(rows), "k": k,
        "allopathy_case_count": len({row["id"] for row in allopathy}),
        "allopathy_primary_strategy": "hybrid" if "hybrid" in strategy_comparison else ("vector-only" if strategy_comparison else None),
        "allopathy_rag_top1_relevance": primary.get("rag_top1_relevance"),
        "allopathy_mean_context_field_coverage": primary.get("mean_context_field_coverage"),
        "allopathy_source_provenance_coverage": primary.get("source_provenance_coverage"),
        "allopathy_strategy_comparison": strategy_comparison,
        "ayurveda_case_count": len(ayurveda),
        "ayurveda_recall_at_k": statistics.mean(row["top_k_hit"] for row in ayurveda) if ayurveda else None,
        "ayurveda_mrr": statistics.mean(row["reciprocal_rank"] for row in ayurveda) if ayurveda else None,
        "ayurveda_provenance_coverage": statistics.mean(row["provenance_preserved"] for row in ayurveda) if ayurveda else None,
        "generation_enabled": generate,
    }
    if generate:
        generated = [row for row in allopathy if "citation_audit" in row]
        summaries = [row["citation_audit"] for row in generated]
        term_scores = [row["answer_term_coverage"] for row in generated if row.get("answer_term_coverage") is not None]
        abstention_cases = [row for row in generated if row.get("expected_abstention")]
        summary.update({
            "generated_answer_count": len(generated),
            "citation_rate": statistics.mean(bool(item["has_citation"]) for item in summaries) if summaries else None,
            "valid_citation_rate": statistics.mean(bool(item["all_citations_resolve"]) for item in summaries) if summaries else None,
            "mean_required_answer_term_coverage": statistics.mean(term_scores) if term_scores else None,
            "expected_abstention_rate": statistics.mean(row["expected_abstention_met"] for row in abstention_cases) if abstention_cases else None,
            "safety_review_flag_count": sum(len(row.get("safety_review_flags", [])) for row in generated),
        })
    return summary


def evaluate(cases_path: Path, report_path: Path, k: int = 5, generate: bool = False, strategy: str = "compare") -> dict[str, Any]:
    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if generate:
        # Fail early before doing retrieval work if no provider key is available.
        client = get_groq_client()
        del client
    engine = GraphQueryEngine(db=Neo4jConnection())
    rows = []
    try:
        # Keep model loading and first-call warm-up out of per-query retrieval latency.
        if any(case["mode"] == "allopathy" for case in cases):
            engine.get_embedding("MEDGraphy retrieval benchmark warm-up")
        for case in cases:
            if case["mode"] == "ayurveda":
                rows.append(evaluate_ayurveda(engine, case, k))
            elif case["mode"] == "allopathy":
                strategies = ("vector-only", "hybrid") if strategy == "compare" else (strategy,)
                for selected_strategy in strategies:
                    rows.append(evaluate_allopathy(engine, case, generate, selected_strategy))
            else:
                raise ValueError(f"Unknown benchmark mode: {case['mode']}")
    finally:
        engine.db.close()
    report = {
        "generated_at": utc_now(), "cases_path": str(cases_path),
        "case_set_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "run_config": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_warmed_before_timing": True,
            "generation_provider": "Groq" if generate else None,
            "generation_model": GROQ_MODEL if generate else None,
            "retrieval_k": k,
            "retrieval_strategy": strategy,
            "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        },
        "retrieval_k": k, "summary": summarize(rows, generate, k), "cases": rows,
        "limitations": [
            "The curated set is small and does not establish clinical accuracy.",
            "Allopathy RAG retrieves one medicine context per query; its top-1 metric reflects that design.",
            "Per-query latency excludes embedding-model initialization and warm-up.",
            "Ayurveda is benchmarked as catalog retrieval only; its imported text remains unverified.",
            "Citation presence and automated safety flags do not prove that claims are supported or safe.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    summary = report["summary"]
    log_event("rag_benchmark_completed", case_count=len(rows), generate=generate, strategy=strategy,
              allopathy_rag_top1_relevance=summary["allopathy_rag_top1_relevance"],
              ayurveda_recall_at_k=summary["ayurveda_recall_at_k"], report_path=str(report_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, default=5, help="Ayurveda retrieval cutoff (Allopathy RAG uses its production top-1 strategy)")
    parser.add_argument("--generate", action="store_true", help="Call the configured Groq model for Allopathy cases; may incur provider usage")
    parser.add_argument("--strategy", choices=("compare", "hybrid", "vector-only"), default="compare", help="Compare both retrievers, or run one strategy")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.generate and args.strategy == "compare":
        parser.error("--generate requires --strategy hybrid or --strategy vector-only to avoid duplicate provider calls")
    report = evaluate(args.cases, args.out, args.k, args.generate, args.strategy)
    print(json.dumps({"report": str(args.out), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
