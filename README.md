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

Run the retrieval benchmark against the local Neo4j graph. It does not call a language model unless `--generate` is supplied:

```bash
.venv/bin/python evaluate_rag.py --split dev
.venv/bin/python evaluate_rag.py --split test --latency-repeats 100 --out reports/rag_benchmark/test.json
```

`--split dev` is for calibrating abstention thresholds. `--split test` evaluates the reserved new probes once; do not tune on those cases and then keep reporting them as held out. The original catalog examples are marked `exploratory` because their results were previously viewed. The development and test probes cover unknown products, far misspellings, non-medicine queries, symptom triage, paraphrased conditions, ingredient search, side effects, manufacturer search, and a multi-constraint query.

The comparator includes vector-only, exact/fuzzy-name-only, Lucene full-text-only, condition-graph-only, hybrid without exact-name priority, and hybrid with exact-name priority. Named-product and other single-answer lookups report Recall@k, MRR, and nDCG@k by query type. Condition and catalog-list tasks report Precision@k and Success@k, rather than blending their large relevant sets into Recall@k. Negative queries report the abstention rate; positive queries report false-abstain rate. Hybrid versus vector-only uses a paired exact McNemar test on shared binary outcomes, described as exploratory.

### Fresh held-out retrieval run

On 2026-11-24, the first run of the fresh `test` split used 21 probes (15 negative, 6 positive), 100 warmed retrieval observations per query, and the local Neo4j catalog. The test cases were authored separately after the earlier test-v1 cases were moved to `dev`; do not tune on this split and continue to call it held out. These results are a small, curated development signal, not a claim of production accuracy.

| Strategy | Negative abstention | False positive results | Positive false abstention | Warm p50 / p95 |
|---|---:|---:|---:|---:|
| Exact/fuzzy only | 14/15 (93.3%) | 1/15 | 5/6 (83.3%) | 16.43 / 33.72 ms |
| Lucene full-text only | 14/15 (93.3%) | 1/15 | 3/6 (50.0%) | 1.17 / 20.97 ms |
| Condition graph only | 15/15 (100%) | 0/15 | 4/6 (66.7%) | 1.46 / 22.00 ms |
| Vector only | 15/15 (100%) | 0/15 | 2/6 (33.3%) | 12.29 / 39.75 ms |
| Hybrid without exact-name priority | 15/15 (100%) | 0/15 | 0/6 | 12.36 / 41.70 ms |
| Hybrid | 15/15 (100%) | 0/15 | 0/6 | 33.44 / 57.89 ms |

The hybrid and hybrid-without-exact-priority each retrieved the labeled result for the one case in every positive category: condition paraphrase, ingredient, side effect, manufacturer, multi-constraint, and a misspelled product name. Each category has only one probe, so these 1/1 results have very wide uncertainty (95% Wilson interval 20.7–100%) and are not broad performance estimates. Hybrid’s 0/6 positive false-abstain result also has a 0–39.0% Wilson interval; the 15/15 negative abstention result has a 79.6–100% interval. For the paired hybrid/vector ranking outcomes, hybrid had 1 win, vector had 0, and 5 tied across 6 positives; exact two-sided McNemar p = 1.00. This is descriptive and not statistically significant. Hybrid’s warmed p95 was about 1.46× vector-only in this run. The JSON report also records the per-query-type latency samples, engine initialization (1,724 ms), first-unwarmed query by strategy, settings, and case-set SHA-256. Each strategy has 2,100 warmed observations (21 cases × 100 repetitions); this is repeated local retrieval timing, not an independent 2,100-query relevance sample or a load test.

Retrieval abstains on recommendation and urgent symptom-triage questions, unknown product-detail requests without a high-confidence name match, unsupported query classes, and vector-only candidates below `RAG_MIN_VECTOR_SIMILARITY` unless lexical or graph evidence supports the match. Fuzzy product matching uses `RAG_FUZZY_PRODUCT_SIMILARITY`. Both thresholds are recorded in the JSON report and can be set in `.env.local`. Abstention is not a clinical triage system.

For timing, the report records engine initialization, one first query per strategy, and warmed p50/p95 with the sample count. `--latency-repeats 100` collects 100 warmed observations per query; per-query-type latency is also reported. These local measurements do not represent production concurrency or deployment latency.

The current curated cases are not a random sample. The original Allopathy cases use stable IDs from the checked-in catalog CSV; condition phrase labels indicate literal dataset matches and do not establish clinical appropriateness. Ayurveda’s 14 cases were manually selected from the local, untracked Ayurveda CSV using known problem, symptom, and herb record IDs. Their 14/14 result is a smoke test; a 95% Wilson interval is 78.5–100%, and the source data is not committed to Git.

The latest benchmark JSON is generated locally under `reports/rag_benchmark/` and is excluded from Git. The README reports only measured splits and states their limits. Generated-answer evaluation remains separate: citation resolution and answer-term coverage do not establish claim-level faithfulness or medical correctness. The recent Groq generation attempt received HTTP 429, so no generation result is published. The lower-level graph checks remain available as `.venv/bin/python evaluate_retrieval.py`.

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
