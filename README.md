# MEDGraphy

MEDGraphy is a medicine catalog and graph retrieval app with Allopathy and Ayurveda modes. FastAPI serves the JSON API and the single-page HTML frontend. Neo4j stores the modern-medicine graph and vector embeddings; Groq Qwen generates grounded answers from retrieved catalog context.

## Safety and data provenance

The checked-in medicine CSV is catalog metadata, not verified clinical guidance. Its upstream publisher, source date, license, and jurisdiction are not documented, so records carry incomplete provenance and the UI displays that status. No interaction claims are inferred from shared ingredients. Interaction records require reviewed evidence and source metadata.

The Ayurveda CSV is imported into separate `AyurvedaEntry` and `AyurvedicHerb` graph labels for data review. Every record remains `unverified_preview`; `MENTIONS_HERB` records a source-row mention and does not mean that an herb treats a condition. The original row is retained as JSON with source-row identity and a file SHA-256. Search the imported graph through the Ayurveda mode. Plant-leaf images are shown only for exact common-name folder matches. The supplied publisher, license, and clinical claims are not verified, so verify source rights and record quality before changing this status or making clinical use. The crosswalk at `data/ayurveda/ayurveda_link_candidates.csv` does not create links between Ayurveda and modern medicine records.

## Local setup

Requirements: Python 3.12 (the project virtual environment uses Python 3.12), Docker Desktop, and a Groq API key for generated answers. Medicine lookup works without a Groq key.

```bash
cp .env.local.example .env.local
# Edit .env.local and set NEO4J_PASSWORD and GROQ_API_KEY.
docker compose --env-file .env.local up -d neo4j
.venv/bin/python -m pip install -r requirements.txt
```

If `.venv` does not exist yet, create it with `python3.12 -m venv .venv` first. `.env.local` is ignored by Git.

## Run the full stack with Docker

The Docker image runs FastAPI and serves the HTML frontend from the same container; Docker Compose starts that app alongside Neo4j. After creating `.env.local` from the template above and setting a Neo4j password, start both services with:

```bash
docker compose --env-file .env.local up --build -d
```

Open [http://localhost:8000](http://localhost:8000). The first app start imports the bundled Allopathy catalog and builds its embeddings if the Neo4j database has no medicine records; this can take several minutes and downloads the embedding model. On later starts it checks/creates the retrieval indexes. Existing Neo4j data is kept in the named Docker volumes. Neo4j Browser remains available at [http://localhost:7474](http://localhost:7474).

The local Ayurveda directory is mounted read-only into the app container, so images and a locally supplied CSV can be used without being copied into the image or Git. Ayurveda records are not automatically ingested; follow the manual audit/import steps below after the app is running:

```bash
docker compose --env-file .env.local exec app python ingest_ayurveda.py --ingest
```

View service status and logs, or stop the services while retaining data:

```bash
docker compose --env-file .env.local ps
docker compose --env-file .env.local logs -f app
docker compose --env-file .env.local down
```

`down` preserves Neo4j volumes. Only `docker compose down -v` deletes the database volumes. Set `AUTO_INGEST=false` in `.env.local` to disable automatic Allopathy import on an empty database.

Local `.env` files are ignored; `.env.local.example` and `.env.aws.example` are configuration templates. The Ayurveda text CSV, optional workbook, and leaf-image files are excluded from Git because of their size. The leaf images can be downloaded from [Medicinal Leaf Dataset — Mendeley Data](https://data.mendeley.com/datasets/nnytj2v3n5/1), which lists a CC BY 4.0 license; see [`data/ayurveda/README.md`](data/ayurveda/README.md) for the exact extraction path and attribution note. That Mendeley link is only for the leaf images; the separate Ayurveda text CSV has no verified upstream download URL or license documented in this repository and must be supplied locally at `data/ayurveda/ayurveda_dataset.csv` for ingestion.

To populate a new local Neo4j database with the medicine catalog and vector embeddings:

```bash
.venv/bin/python ingest_graph.py --csv data/Medicine_Details.csv
```

The ingestion command creates the vector and full-text indexes. If the medicine graph was already populated before hybrid retrieval was added, create its full-text index once with `.venv/bin/python ensure_rag_indexes.py`.

Audit the Ayurveda file first, then import its rows into isolated graph labels. The import is additive and does not clear the database:

```bash
.venv/bin/python ingest_ayurveda.py
.venv/bin/python ingest_ayurveda.py --ingest
```

Use `--limit 100` with `--ingest` for a small staging import. The audit writes `reports/ayurveda/data_quality_report.json` and `duplicate_review.csv`. The source manifest is `data/ayurveda/source_manifest.json`; fill in the actual upstream source URL, retrieval date, jurisdiction, and license only after verification. To refresh records after reviewing/updating the CSV, run the import again; stable IDs make it upsert the corresponding entries.

To refresh catalog-managed relationships without deleting the database, add `--rebuild-managed-relationships`. Use `--clear` only when intentionally replacing the entire graph. Existing local data is stored in Docker volumes `medgraphy_neo4j_data` and `medgraphy_neo4j_logs`.

Start the app:

```bash
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is at `/docs`; health is at `/api/health`.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Database, vector-index, and full-text-index status |
| `GET` | `/api/medicines/search?q=...` | Search catalog products by name |
| `GET` | `/api/medicines/{medicine_id}` | Retrieve one product record and provenance |
| `GET` | `/api/conditions/search?q=...` | Find catalog products with matching condition text, with conservative spelling correction when exact lookup misses |
| `GET` | `/api/medicines/{medicine_id}/shared-ingredients` | Show ingredient overlaps, explicitly separate from interactions |
| `GET` | `/api/medicines/{medicine_id}/interactions` | Return only evidence-backed interaction records |
| `POST` | `/api/rag` | Retrieve catalog context and generate an answer with source metadata; accepts up to eight recent chat messages and an active medicine ID for follow-up questions |
| `GET` | `/api/ayurveda/search?q=...` | Search imported Ayurveda graph records in preview mode |

Example RAG request:

```bash
curl -X POST http://127.0.0.1:8000/api/rag \
  -H 'Content-Type: application/json' \
  -d '{"query":"What ingredients are listed for Augmentin 625 Duo Tablet?"}'
```

## Data quality and ingestion

Run the catalog audit and duplicate review:

```bash
.venv/bin/python data_pipeline.py --csv data/Medicine_Details.csv
```

It writes `reports/data_quality/data_quality_report.json` and `duplicate_review.csv`. Product IDs are stable hashes of normalized name, composition, and manufacturer. Original source values, row IDs, source hash, and duplicate variants are retained in Neo4j.

The local interaction intake template is `data/interactions_template.csv`. Only ingest reviewed records with endpoint medicine IDs, evidence source and HTTPS URL, evidence summary, jurisdiction, review date, license, severity, and provenance status:

```bash
.venv/bin/python ingest_interactions.py data/interactions.csv
```

The importer rejects unreviewed or insufficiently sourced rows. No interaction dataset is included by default.

## Evaluation and monitoring

Run the RAG benchmark against the local Neo4j graph. The default run compares vector-only retrieval with hybrid retrieval on the same Allopathy cases, and measures Ayurveda search separately. It stays local and does not call an LLM provider:

```bash
.venv/bin/python evaluate_rag.py
```

It reports Allopathy and Ayurveda retrieval with the same ranking metrics (Hit@k, Recall@k, MRR, and nDCG@k), broken down by query type. Allopathy labels resolve to stable medicine IDs from the checked-in catalog. Condition-query labels identify records whose catalog `Uses` field contains the requested phrase; this measures catalog text retrieval, not clinical appropriateness. Two negative cases measure whether out-of-scope advice and unknown product queries return any catalog result. A returned record is counted as a false positive for those cases.

Hybrid retrieval first checks exact product names. Otherwise it fuses condition graph matches, Neo4j Lucene full-text results, and vector candidates with weighted Reciprocal Rank Fusion (condition matches weight 2, RRF constant 60). The report includes per-case ranked candidates so query-type results can be inspected.

The benchmark reports a Wilson 95% interval for binary Hit@k and p50/p95 latency across the timed retrieval samples. For 100 timed retrievals per query, run:

```bash
.venv/bin/python evaluate_rag.py --latency-repeats 100 --out reports/rag_benchmark/latest.json
```

Latency is measured after the embedding model is loaded and warmed; it is not a cold-start or production-load benchmark. The number of measured samples is included in the JSON report. Compare retrieval methods by query type because exact product-name lookups are a different task from condition search. The candidate ranking list is exposed in the report, while the current answer-generation path still supplies one medicine context.

`eval/rag_cases.json` currently contains 41 Allopathy and 14 Ayurveda cases. They are exploratory development cases, not a held-out test set; previous retrieval results were inspected while developing this benchmark. Do not describe them as independent generalization results. Expand the reviewed test split after tuning on a separate development set. Ayurveda labels validate record lookup and provenance retention only; the source remains unverified.

Latest local retrieval run (2026-11-24; one warmed run with 100 repeats per Allopathy query, 4,100 latency samples per strategy; generation disabled):

| Query type / strategy | Cases | Hit@5 (95% Wilson CI) | Recall@5 | MRR | nDCG@5 | p50 / p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Named product — Hybrid | 29 | 29/29 (88.3–100%) | 100% | 1.00 | 1.00 | — |
| Named product — Vector-only | 29 | 25/29 (69.4–94.5%) | 86.2% | 0.77 | 0.79 | — |
| Catalog condition phrase — Hybrid | 10 | 10/10 (72.2–100%) | 2.2% | 1.00 | 1.00 | — |
| Catalog condition phrase — Vector-only | 10 | 9/10 (59.6–98.2%) | 1.4% | 0.90 | 0.79 | — |
| All labeled queries — Hybrid | 39 | 39/39 (91.0–100%) | 74.9% | 1.00 | 1.00 | 21.3 / 55.0 ms |
| All labeled queries — Vector-only | 39 | 34/39 (73.3–94.4%) | 64.5% | 0.80 | 0.79 | 18.4 / 24.8 ms |
| Out-of-scope + unknown product — both strategies | 2 | Abstained 0/2 | — | — | — | included above |
| Ayurveda catalog lookup | 14 | 14/14 | 100% | 1.00 | 1.00 | — |

This small sample does not show a statistically clear difference between the two strategies: their aggregate Hit@5 Wilson intervals overlap. Hybrid’s named-product cases are exact-name lookups, and it returned irrelevant catalog records for both negative queries. The condition metrics measure retrieved catalog rows, not whether a medicine is clinically appropriate for a condition. Ayurveda remains a smoke test. The JSON report is generated locally at `reports/rag_benchmark/latest.json` and is intentionally excluded from Git; rerun the command above to reproduce against your local graph and data.

Ayurveda source provenance was retained for all 14 retrieved records in this run. This checks that source identifiers and import status survived ingestion; it does not validate the source or its claims.

To include answer generation, citations, expected answer terms, abstention cases, and automated safety review flags, use the configured Groq model:

```bash
.venv/bin/python evaluate_rag.py --generate --strategy hybrid --out reports/rag_benchmark/qwen.json
```

Generation calls can incur provider usage. The harness reports citation resolution and expected-term coverage, but neither is a claim-level faithfulness metric. Claim-level support and medical correctness require human-reviewed annotations. Automated safety flags are review signals, not clinical judgments. The lower-level graph checks remain available as `.venv/bin/python evaluate_retrieval.py`.

Operational events are appended to `runtime/medgraph_events.jsonl`. The logger records ingestion outcomes, retrieval hits/misses, citation audits, interaction imports, and evaluation scores without storing raw query text or generated answers. Missing dates remain `unknown` rather than being inferred.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` in the example file | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | required | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database |
| `GROQ_API_KEY` | optional for lookup | Enables generated answers |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq chat model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `VECTOR_INDEX_NAME` | `medicine_embeddings` | Neo4j vector index name |
| `FULLTEXT_INDEX_NAME` | `medicine_catalog_fulltext` | Neo4j Lucene full-text index name for hybrid retrieval |

Stop the local database with `docker compose --env-file .env.local stop neo4j`. This keeps the named data volumes intact.
