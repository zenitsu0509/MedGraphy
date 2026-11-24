"""Idempotently create the Neo4j vector and full-text indexes used by RAG."""

from __future__ import annotations

from graph_db import Neo4jConnection
from graph_rag_query import FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME
from ingest_graph import VECTOR_INDEX_CYPHER


def main() -> None:
    database = Neo4jConnection()
    try:
        database.query(VECTOR_INDEX_CYPHER, db="neo4j")
        vector_rows = database.query(
            "SHOW VECTOR INDEXES YIELD name, state WHERE name=$name RETURN state",
            {"name": VECTOR_INDEX_NAME}, db="neo4j",
        )
        vector_state = vector_rows[0]["state"] if vector_rows else "missing"
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
        print(f"Vector index {VECTOR_INDEX_NAME}: {vector_state}")
        print(f"Full-text index {FULLTEXT_INDEX_NAME}: {state}")
    finally:
        database.close()


if __name__ == "__main__":
    main()
