#!/bin/sh
set -eu

python - <<'PY'
import os
import subprocess
import sys

from graph_db import Neo4jConnection

db = Neo4jConnection()
try:
    rows = db.query("MATCH (m:Medicine) RETURN count(m) AS count", db="neo4j")
    medicine_count = rows[0]["count"] if rows else 0
finally:
    db.close()

if medicine_count == 0:
    if os.getenv("AUTO_INGEST", "true").lower() not in {"1", "true", "yes"}:
        raise SystemExit("Neo4j has no medicine records. Set AUTO_INGEST=true or run ingest_graph.py.")
    print("No medicine records found; importing the bundled catalog and building embeddings.", flush=True)
    subprocess.run([sys.executable, "ingest_graph.py", "--csv", "data/Medicine_Details.csv"], check=True)
else:
    print(f"Found {medicine_count} medicine records; checking retrieval indexes.", flush=True)
    subprocess.run([sys.executable, "ensure_rag_indexes.py"], check=True)

os.execvp("uvicorn", ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])
PY
