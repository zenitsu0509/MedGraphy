"""Idempotently create the Neo4j full-text index used by hybrid RAG."""

from __future__ import annotations

from graph_db import Neo4jConnection
from graph_rag_query import FULLTEXT_INDEX_NAME


def main() -> None:
    database = Neo4jConnection()
    try:
        database.query(
            f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
            "FOR (m:Medicine) ON EACH [m.name, m.composition, m.uses_text, m.side_effects_text]",
            db="neo4j",
        )
        rows = database.query(
            "SHOW FULLTEXT INDEXES YIELD name, state WHERE name=$name RETURN state",
            {"name": FULLTEXT_INDEX_NAME}, db="neo4j",
        )
        state = rows[0]["state"] if rows else "missing"
        print(f"Full-text index {FULLTEXT_INDEX_NAME}: {state}")
    finally:
        database.close()


if __name__ == "__main__":
    main()
