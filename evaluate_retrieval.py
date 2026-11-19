"""Run lightweight graph retrieval checks and report Recall@k, MRR, and unsupported claims."""
import argparse
import json
from pathlib import Path

from graph_db import Neo4jConnection
from graph_rag_query import GraphQueryEngine
from monitoring import log_event


def evaluate(cases_path: str, k: int = 5) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]
    engine = GraphQueryEngine.__new__(GraphQueryEngine)
    engine.db = Neo4jConnection()
    rows = []
    try:
        for case in cases:
            route, query = case["route"], case["query"]
            expected = set(case["expected"])
            if route == "medicine":
                results = engine.db.query("""
                    MATCH (m:Medicine)
                    WHERE toLower(m.name)=toLower($name)
                    RETURN m.name AS name ORDER BY m.name LIMIT $limit
                """, {"name": query, "limit": k}, db="neo4j") or []
                found = [item["name"] for item in results]
            elif route == "condition":
                found = engine.reverse_lookup(query)[:k]
            elif route == "known_interaction":
                found = [item["other_medicine"] for item in engine.known_interactions(query)[:k]]
            else:
                raise ValueError(f"Unknown evaluation route: {route}")
            ranks = [index + 1 for index, item in enumerate(found) if item in expected]
            rows.append({"id": case["id"], "route": route, "expected": sorted(expected), "found": found,
                         "recall_at_k": len(set(found) & expected) / len(expected) if expected else (1.0 if not found else 0.0),
                         "reciprocal_rank": (1 / min(ranks)) if ranks else 0.0,
                         "unsupported_claims": 0 if route != "known_interaction" or not found else len(found)})
    finally:
        engine.db.close()
    count = len(rows) or 1
    report = {"case_count": len(rows), "k": k, "recall_at_k": sum(row["recall_at_k"] for row in rows) / count,
              "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
              "unsupported_interaction_claims": sum(row["unsupported_claims"] for row in rows), "cases": rows}
    log_event("retrieval_evaluation", case_count=len(rows), recall_at_k=report["recall_at_k"], mrr=report["mrr"], unsupported_interaction_claims=report["unsupported_interaction_claims"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="eval/retrieval_cases.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cases, args.k), indent=2))
