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
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipeline import normalize_key
from graph_db import Neo4jConnection
from graph_rag_query import EMBEDDING_MODEL, FUZZY_PRODUCT_SIMILARITY, MIN_VECTOR_SIMILARITY, GraphQueryEngine
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


def rank_metrics(ranked_ids: list[str], expected_ids: set[str], k: int) -> dict[str, float]:
    """Compute standard ranking metrics using stable catalog medicine IDs."""
    relevant_ranks = [rank for rank, medicine_id in enumerate(ranked_ids[:k], 1) if medicine_id in expected_ids]
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(expected_ids), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "hit_at_k": float(bool(relevant_ranks)),
        "success_at_k": float(bool(relevant_ranks)),
        "precision_at_k": len(set(ranked_ids[:k]) & expected_ids) / k if expected_ids else 0.0,
        "recall_at_k": len(set(ranked_ids[:k]) & expected_ids) / len(expected_ids) if expected_ids else 0.0,
        "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def wilson_interval(successes: int, count: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval for a binary rate; descriptive for small samples."""
    if count == 0:
        return None
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) / count) + (z * z / (4 * count * count))) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def exact_mcnemar_p(hybrid_only: int, vector_only: int) -> float | None:
    """Two-sided exact McNemar p-value from discordant paired binary outcomes."""
    discordant = hybrid_only + vector_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(hybrid_only, vector_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def case_metric_kind(case: dict[str, Any]) -> str:
    return case.get("metric_kind") or (
        "abstention" if case.get("expected_no_catalog_match") else
        "condition_topk" if str(case.get("query_type", "")).startswith("condition") or case.get("query_type") == "catalog_condition_phrase" else
        "single_answer"
    )


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
    recall = len(set(found_ids[:k]) & expected) / len(expected) if expected else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return {
        "id": case["id"], "mode": "ayurveda", "query": case["query"], "field": field,
        "expected_ids": sorted(expected), "found_ids": found_ids,
        "found_labels": [row["problem"] for row in rows],
        "top_k_hit": matched, "recall_at_k": recall, "reciprocal_rank": 1 / min(ranks) if ranks else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "source_sha256_values": sorted({row.get("source_sha256") for row in rows if row.get("source_sha256")}),
        "provenance_preserved": bool(rows) and all(row["provenance_status"] == "unverified_preview" and row["source_id"] for row in rows),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def evaluate_allopathy(engine: GraphQueryEngine, case: dict[str, Any], generate: bool, strategy: str, k: int, latency_repeats: int) -> dict[str, Any]:
    latency_samples = []
    retrieved = None
    for _ in range(max(1, latency_repeats)):
        started = time.perf_counter()
        retrieved = engine.retrieve_context_for_rag(case["query"], strategy=strategy)
        latency_samples.append((time.perf_counter() - started) * 1000)
    retrieval_ms = latency_samples[0]
    context = (retrieved or {}).get("context") or {}
    name = (retrieved or {}).get("medicine_found")
    expected = set(case.get("expected_medicines", []))
    matched_facts = {
        field: phrase for field, phrase in case.get("expected_context_contains", {}).items()
        if normalize_key(phrase) in normalize_key(context.get(field, ""))
    }
    all_facts_matched = len(matched_facts) == len(case.get("expected_context_contains", {}))
    ranked_candidates = (retrieved or {}).get("ranked_candidates", [])
    ranked_names = [item.get("name", "") for item in ranked_candidates]
    expected_ids = set(case.get("expected_medicine_ids", []))
    if not expected_ids:
        expected_ids = {normalize_key(name) for name in expected}
    ranked_ids = [item.get("medicine_id") or normalize_key(item.get("name", "")) for item in ranked_candidates]
    metrics = rank_metrics(ranked_ids, expected_ids, k)
    expected_abstention = bool(case.get("expected_no_catalog_match"))
    query_type = case.get("query_type") or ("condition" if case.get("expected_context_contains") else ("named_product" if expected else "out_of_scope"))
    metric_kind = case_metric_kind(case)
    required_fields = case.get("required_context_fields", [])
    missing_fields = [field for field in required_fields if not context.get(field)]
    result: dict[str, Any] = {
        "id": case["id"], "mode": "allopathy", "query": case["query"],
        "query_type": query_type, "split": case.get("split", "exploratory"),
        "metric_kind": metric_kind,
        "strategy": strategy,
        "expected_medicines": sorted(expected), "found_medicine": name,
        "expected_medicine_ids": sorted(case.get("expected_medicine_ids", [])),
        "label_source": case.get("label_source", "curator_case_label"),
        "relevance_basis": "medicine_id" if case.get("expected_medicine_ids") else ("normalized_exact_medicine_name" if expected else "unlabeled"),
        "expected_context_contains": case.get("expected_context_contains", {}),
        "matched_context_facts": matched_facts,
        **metrics,
        "ranked_medicines": ranked_names[:k], "ranked_medicine_ids": ranked_ids[:k],
        "context_phrase_match": bool(case.get("expected_context_contains")) and all_facts_matched,
        "expected_catalog_abstention": expected_abstention,
        "abstention_detected": not bool(retrieved) or bool((retrieved or {}).get("abstained")),
        "false_positive_result": expected_abstention and bool(retrieved) and not bool((retrieved or {}).get("abstained")),
        "false_abstain_result": bool(expected_ids) and (not bool(retrieved) or bool((retrieved or {}).get("abstained"))),
        "context_fields_missing": missing_fields,
        "context_field_coverage": (len(required_fields) - len(missing_fields)) / len(required_fields) if required_fields else 1.0,
        "source_id": context.get("source_id"),
        "source_sha256": context.get("source_sha256"),
        "provenance_status": context.get("provenance_status"),
        "retrieval_sources": context.get("retrieval_sources", []),
        "rrf_score": context.get("retrieval_score"),
        "top_vector_similarity": (retrieved or {}).get("top_vector_similarity"),
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "retrieval_latency_samples_ms": [round(value, 2) for value in latency_samples],
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
            "expected_abstention": bool(case.get("expected_abstention") or case.get("expected_no_catalog_match")),
            "abstention_detected": bool(ABSTENTION.search(answer)) if (case.get("expected_abstention") or case.get("expected_no_catalog_match")) else None,
            "expected_abstention_met": bool(ABSTENTION.search(answer)) if (case.get("expected_abstention") or case.get("expected_no_catalog_match")) else None,
            "safety_review_flags": safety_flags,
        })
    elif generate:
        should_abstain = bool(case.get("expected_abstention") or case.get("expected_no_catalog_match"))
        result.update({
            "answer": None, "citation_audit": audit_citations("", []), "safety_review_flags": [],
            "expected_abstention": should_abstain,
            "abstention_detected": should_abstain,
            "expected_abstention_met": should_abstain,
        })
    return result


def summarize(rows: list[dict[str, Any]], generate: bool, k: int) -> dict[str, Any]:
    allopathy = [row for row in rows if row["mode"] == "allopathy"]
    ayurveda = [row for row in rows if row["mode"] == "ayurveda"]
    strategy_comparison = {}
    for strategy in sorted({row["strategy"] for row in allopathy}):
        subset = [row for row in allopathy if row["strategy"] == strategy]
        by_type = {}
        for query_type in sorted({row["query_type"] for row in subset}):
            typed = [row for row in subset if row["query_type"] == query_type]
            latency_samples = [sample for row in typed for sample in row.get("retrieval_latency_samples_ms", [])]
            metric_kind = typed[0]["metric_kind"]
            if metric_kind == "abstention":
                abstained = sum(row["abstention_detected"] for row in typed)
                by_type[query_type] = {
                    "metric_kind": metric_kind, "case_count": len(typed),
                    "abstention_rate": abstained / len(typed) if typed else None,
                    "false_positive_results": len(typed) - abstained,
                    "latency_sample_count": len(latency_samples),
                    "latency_p50_ms": percentile(latency_samples, .50), "latency_p95_ms": percentile(latency_samples, .95),
                }
                continue
            positive = [row for row in typed if row["relevance_basis"] != "unlabeled"]
            matched = sum(row["hit_at_k"] for row in positive)
            metrics = {
                "metric_kind": metric_kind, "case_count": len(positive),
                "success_at_k": matched / len(positive) if positive else None,
                "success_at_k_95ci_wilson": wilson_interval(int(matched), len(positive)),
                "false_abstain_rate": statistics.mean(row["false_abstain_result"] for row in positive) if positive else None,
                "latency_sample_count": len(latency_samples),
                "latency_p50_ms": percentile(latency_samples, .50), "latency_p95_ms": percentile(latency_samples, .95),
            }
            if metric_kind == "condition_topk":
                metrics["precision_at_k"] = statistics.mean(row["precision_at_k"] for row in positive) if positive else None
            elif metric_kind == "catalog_topk":
                metrics["precision_at_k"] = statistics.mean(row["precision_at_k"] for row in positive) if positive else None
            else:
                metrics.update({
                    "recall_at_k": statistics.mean(row["recall_at_k"] for row in positive) if positive else None,
                    "mrr": statistics.mean(row["mrr"] for row in positive) if positive else None,
                    "ndcg_at_k": statistics.mean(row["ndcg_at_k"] for row in positive) if positive else None,
                })
            by_type[query_type] = metrics
        latency_samples = [sample for row in subset for sample in row.get("retrieval_latency_samples_ms", [])]
        negatives = [row for row in subset if row.get("expected_catalog_abstention")]
        positives = [row for row in subset if row["relevance_basis"] != "unlabeled"]
        first_query_rows = [row for row in subset if row.get("first_query_latency_ms") is not None]
        strategy_comparison[strategy] = {
            "case_count": len(subset),
            "query_type_metrics": by_type,
            "positive_case_count": len(positives),
            "false_abstain_rate": statistics.mean(row["false_abstain_result"] for row in positives) if positives else None,
            "negative_case_count": len(negatives),
            "negative_abstention_rate": statistics.mean(row["abstention_detected"] for row in negatives) if negatives else None,
            "false_positive_results": sum(row["false_positive_result"] for row in negatives),
            "mean_context_field_coverage": statistics.mean(row["context_field_coverage"] for row in positives) if positives else None,
            "source_provenance_coverage": statistics.mean(bool(row["source_id"] and row["provenance_status"]) for row in positives) if positives else None,
            "latency_sample_count": len(latency_samples),
            "first_query_latency_ms": first_query_rows[0]["first_query_latency_ms"] if first_query_rows else None,
            "retrieval_latency_p50_ms": percentile(latency_samples, 0.50),
            "retrieval_latency_p95_ms": percentile(latency_samples, 0.95),
        }
    paired = {}
    left = {row["id"]: row for row in allopathy if row["strategy"] == "hybrid" and row["relevance_basis"] != "unlabeled"}
    right = {row["id"]: row for row in allopathy if row["strategy"] == "vector-only" and row["relevance_basis"] != "unlabeled"}
    shared = sorted(left.keys() & right.keys())
    if shared:
        hybrid_only = sum(left[key]["hit_at_k"] > right[key]["hit_at_k"] for key in shared)
        vector_only = sum(right[key]["hit_at_k"] > left[key]["hit_at_k"] for key in shared)
        paired = {
            "comparison": "hybrid_vs_vector-only", "paired_case_count": len(shared),
            "hybrid_only_wins": hybrid_only, "vector_only_wins": vector_only,
            "ties": len(shared) - hybrid_only - vector_only,
            "exact_mcnemar_p_two_sided": exact_mcnemar_p(hybrid_only, vector_only),
            "interpretation": "descriptive; exploratory case set, not a confirmatory significance claim",
        }
    summary: dict[str, Any] = {
        "case_count": len(rows), "k": k,
        "allopathy_case_count": len({row["id"] for row in allopathy}),
        "allopathy_primary_strategy": "hybrid" if "hybrid" in strategy_comparison else ("vector-only" if strategy_comparison else None),
        "allopathy_strategy_comparison": strategy_comparison,
        "paired_hybrid_vs_vector": paired,
        "ayurveda_case_count": len(ayurveda),
        "ayurveda_hit_at_k": statistics.mean(row["top_k_hit"] for row in ayurveda) if ayurveda else None,
        "ayurveda_hit_at_k_95ci_wilson": wilson_interval(sum(row["top_k_hit"] for row in ayurveda), len(ayurveda)),
        "ayurveda_recall_at_k": statistics.mean(row["recall_at_k"] for row in ayurveda) if ayurveda else None,
        "ayurveda_mrr": statistics.mean(row["reciprocal_rank"] for row in ayurveda) if ayurveda else None,
        "ayurveda_ndcg_at_k": statistics.mean(row["ndcg_at_k"] for row in ayurveda) if ayurveda else None,
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


def evaluate(cases_path: Path, report_path: Path, k: int = 5, generate: bool = False, strategy: str = "compare", latency_repeats: int = 1, split: str = "all") -> dict[str, Any]:
    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if split != "all":
        cases = [case for case in cases if case.get("split", "exploratory") == split]
    if not cases:
        raise ValueError(f"No cases found for split {split!r}")
    engine_started = time.perf_counter()
    if generate:
        # Fail early before doing retrieval work if no provider key is available.
        client = get_groq_client()
        del client
    engine = GraphQueryEngine(db=Neo4jConnection())
    engine_init_ms = round((time.perf_counter() - engine_started) * 1000, 2)
    rows = []
    first_query_by_strategy = {}
    first_case = None
    try:
        if any(case["mode"] == "allopathy" for case in cases):
            allopathy_cases = [case for case in cases if case["mode"] == "allopathy"]
            first_case = next((case for case in allopathy_cases if case_metric_kind(case) == "single_answer"), allopathy_cases[0])
            selected_strategies = ("vector-only", "exact-fuzzy-only", "fulltext-only", "condition-only", "hybrid-no-exact", "hybrid") if strategy == "compare" else (strategy,)
            for selected_strategy in selected_strategies:
                cold_started = time.perf_counter()
                engine.retrieve_context_for_rag(first_case["query"], strategy=selected_strategy)
                first_query_by_strategy[selected_strategy] = round((time.perf_counter() - cold_started) * 1000, 2)
            # Model is loaded at engine initialization; these timings include first query embedding/cache effects.
            engine.get_embedding("MEDGraphy retrieval benchmark warm-up")
        for case in cases:
            if case["mode"] == "ayurveda":
                rows.append(evaluate_ayurveda(engine, case, k))
            elif case["mode"] == "allopathy":
                strategies = ("vector-only", "exact-fuzzy-only", "fulltext-only", "condition-only", "hybrid-no-exact", "hybrid") if strategy == "compare" else (strategy,)
                for selected_strategy in strategies:
                    evaluated = evaluate_allopathy(engine, case, generate, selected_strategy, k, latency_repeats)
                    if first_case and case["id"] == first_case["id"]:
                        evaluated["first_query_latency_ms"] = first_query_by_strategy.get(selected_strategy)
                    rows.append(evaluated)
            else:
                raise ValueError(f"Unknown benchmark mode: {case['mode']}")
    finally:
        engine.db.close()
    report = {
        "generated_at": utc_now(), "cases_path": str(cases_path),
        "case_set_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "run_config": {
            "embedding_model": EMBEDDING_MODEL,
            "min_vector_similarity": MIN_VECTOR_SIMILARITY,
            "fuzzy_product_similarity": FUZZY_PRODUCT_SIMILARITY,
            "embedding_warmed_before_timing": True,
            "engine_initialization_ms": engine_init_ms,
            "first_unwarmed_query_latency_by_strategy_ms": first_query_by_strategy,
            "generation_provider": "Groq" if generate else None,
            "generation_model": GROQ_MODEL if generate else None,
            "retrieval_k": k,
            "retrieval_strategy": strategy,
            "case_split": split,
            "latency_repeats_per_query": latency_repeats,
            "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        },
        "retrieval_k": k, "summary": summarize(rows, generate, k), "cases": rows,
        "limitations": [
            f"The selected {split!r} case split is small and curated; it does not establish clinical accuracy or broad real-world performance.",
            "Allopathy labels use stable medicine IDs; condition IDs are derived from literal phrase matches in the catalog Uses field and remain exploratory labels.",
            "The Allopathy response uses one context record; ranking metrics reflect the available candidate list and may be degenerate for exact-name lookups.",
            "Engine initialization, first-query timings, and warmed p50/p95 are separate local measurements; they are not a production load test.",
            "Ayurveda is benchmarked as catalog retrieval only; its imported text remains unverified.",
            "Citation presence and automated safety flags do not prove that claims are supported or safe.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    summary = report["summary"]
    log_event("rag_benchmark_completed", case_count=len(rows), generate=generate, strategy=strategy,
              allopathy_strategy_count=len(summary["allopathy_strategy_comparison"]),
              ayurveda_recall_at_k=summary["ayurveda_recall_at_k"], report_path=str(report_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, default=5, help="Ayurveda retrieval cutoff (Allopathy RAG uses its production top-1 strategy)")
    parser.add_argument("--latency-repeats", type=int, default=1, help="Measure each Allopathy query this many times after model warm-up")
    parser.add_argument("--split", choices=("all", "dev", "test", "exploratory"), default="all", help="Evaluate a named case split; existing benchmark cases remain exploratory")
    parser.add_argument("--generate", action="store_true", help="Call the configured Groq model for Allopathy cases; may incur provider usage")
    parser.add_argument("--strategy", choices=("compare", "hybrid", "hybrid-no-exact", "vector-only", "fulltext-only", "condition-only", "exact-fuzzy-only"), default="compare", help="Compare retrieval baselines, or run one strategy")
    args = parser.parse_args()
    if args.k < 1 or args.latency_repeats < 1:
        parser.error("--k and --latency-repeats must be positive")
    if args.generate and args.strategy == "compare":
        parser.error("--generate requires one specific retrieval strategy to avoid duplicate provider calls")
    report = evaluate(args.cases, args.out, args.k, args.generate, args.strategy, args.latency_repeats, args.split)
    print(json.dumps({"report": str(args.out), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
