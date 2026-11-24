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
| `POST` | `/api/rag` | Retrieve catalog context and generate an answer with source metadata |
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

It reports Allopathy top-1 relevance, context/provenance coverage, and mean retrieval latency for each strategy, plus Ayurveda Recall@k/MRR and provenance retention. The hybrid path gives exact product-name matches priority; otherwise it fuses graph condition matches, Neo4j Lucene full-text results, and vector candidates with weighted Reciprocal Rank Fusion (condition matches weight 2, RRF constant 60). The curated cases are in `eval/rag_cases.json`; output defaults to `reports/rag_benchmark/latest.json`. Run one strategy with `--strategy hybrid` or `--strategy vector-only`.

Latest local retrieval-only benchmark (six Allopathy cases and three Ayurveda cases, warmed embedding model):

| Strategy | Allopathy top-1 relevance | Context field coverage | Source metadata coverage | Mean retrieval latency |
|---|---:|---:|---:|---:|
| Vector-only | 3/6 (50%) | 6/6 (100%) | 6/6 (100%) | 47.0 ms |
| Hybrid | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 44.4 ms |
| Ayurveda search, Recall@5 | 3/3 (100%) | — | 3/3 (100%) | MRR 1.00 |

This is a small development benchmark against the current local graph; the condition cases count a result as relevant when its retrieved catalog text contains the expected use phrase. It measures retrieval only, not answer faithfulness, medical correctness, or production-scale latency. Expand the reviewed case set before making broad performance claims.

To include answer generation, citations, expected answer terms, abstention cases, and automated safety review flags, use the configured Groq model:

```bash
.venv/bin/python evaluate_rag.py --generate --out reports/rag_benchmark/qwen.json
```

This sends benchmark prompts and retrieved catalog records to Groq and may incur provider usage. Automated safety flags are review signals, not clinical judgments. Citation presence does not prove a claim is supported. Expand and maintain the expected records through curator review; benchmark scores measure this small dataset-specific set and do not establish clinical accuracy. The lower-level retrieval checks remain available as `.venv/bin/python evaluate_retrieval.py`.

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
