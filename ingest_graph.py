import os
import re
import argparse
import hashlib
import json
import re
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv
from data_pipeline import DEFAULT_MANIFEST, load_manifest, normalize_key, prepare_records, stable_medicine_id
from monitoring import log_event

# Use the local Docker profile when present, then fill remaining settings from .env.
load_dotenv(".env.local")
load_dotenv()

# Default embedding model (384-dim)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "medicine_embeddings")
if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", VECTOR_INDEX_NAME):
    raise ValueError("VECTOR_INDEX_NAME must be a simple Neo4j identifier")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME", "medicine_catalog_fulltext")
if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", FULLTEXT_INDEX_NAME):
    raise ValueError("FULLTEXT_INDEX_NAME must be a simple Neo4j identifier")


def parse_active_ingredients(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [p.strip() for p in raw.split('+') if p.strip()]
    cleaned = []
    for p in parts:
        # Remove dosage in parentheses e.g. (500mg) or (0.1% w/w)
        c = re.sub(r"\([^)]*\)", "", p)
        c = re.sub(r"\s+", " ", c).strip()
        if c:
            cleaned.append(c)
    return cleaned

SIDE_EFFECT_END_WORDS = {"pain","bleeding","change","headache","nosebleeds","skin","pressure","protein","urine","inflammation","rash","injury","nausea","diarrhea","insomnia","weight","loss","vomiting","candidiasis","cramps","drowsiness","dizziness","constipation","flatulence","indigestion","heartburn","appetite","weakness","fatigue","fever","redness","swelling","irritation","itching","tremors","palpitations","photophobia","cramp","burn"}

def parse_side_effects(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    # Prefer comma separation if present
    if ',' in raw:
        items = [i.strip() for i in raw.split(',') if i.strip()]
        return items
    tokens = raw.split()
    phrases = []
    current = []
    for idx, tok in enumerate(tokens):
        is_cap = tok[0].isupper()
        if not current:
            current.append(tok)
            continue
        # If token starts capital and previous phrase seems complete -> start new phrase
        if is_cap and (current[-1].lower() in SIDE_EFFECT_END_WORDS):
            phrases.append(' '.join(current))
            current = [tok]
        else:
            current.append(tok)
    if current:
        phrases.append(' '.join(current))
    # Basic cleanup (dedupe, strip)
    cleaned = []
    seen = set()
    for p in phrases:
        c = p.strip(' .;:').strip()
        if c and c.lower() not in seen:
            cleaned.append(c)
            seen.add(c.lower())
    return cleaned

CONDITION_KEYWORDS = [
    "cancer","infection","infections","disease","pain","ulcer","reflux","hypertension","asthma","copd","deficiency","migraine","depression","angina","diarrhea","anxiety","allergic","allergies","dermatitis","fissure","cholesterol","osteoporosis","anemia","epilepsy","tuberculosis","heart failure","anal fissure","vitamin","fever"
]

def extract_conditions(uses_text: str) -> list[str]:
    if not isinstance(uses_text, str) or not uses_text.strip():
        return []
    text = uses_text.replace('Treatment of', ' ').replace('Treatment and prevention of', ' ')
    text = re.sub(r"\s+", " ", text)
    candidates = []
    # Split by two or more spaces or periods if present
    splits = re.split(r"[.;]", text)
    for s in splits:
        s = s.strip()
        if not s:
            continue
        # Heuristic: break into phrases containing a condition keyword
        for kw in CONDITION_KEYWORDS:
            if kw.lower() in s.lower():
                candidates.append(s)
                break
    # Additional splitting for multi-condition strings (e.g. multiple cancers)
    refined = []
    for c in candidates:
        # Attempt to separate multiple conditions by ' cancer' etc
        if ' cancer' in c.lower():
            parts = re.split(r"(?i)cancer", c)
            tmp = []
            for p in parts[:-1]:
                p = p.strip(' ,;')
                if p:
                    tmp.append(p + ' cancer')
            last_tail = parts[-1].strip()
            if last_tail:
                tmp.append(last_tail)
            if tmp:
                refined.extend(tmp)
                continue
        refined.append(c)
    # Normalize: Title case, trim
    norm = []
    seen = set()
    for r in refined:
        c = re.sub(r"\s+", " ", r).strip(' ,')
        # Remove leading generic words
        c = re.sub(r"^(the |a |an )", "", c, flags=re.I)
        if len(c) < 3:
            continue
        key = c.lower()
        if key not in seen:
            seen.add(key)
            norm.append(c)
    return norm[:12]  # limit to avoid explosion

def get_driver():
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME')
    pwd = os.getenv('NEO4J_PASSWORD')
    if not all([uri, user, pwd]):
        raise ValueError("Missing Neo4j credentials in environment (.env)")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    return driver

SCHEMA_QUERIES = [
    "DROP CONSTRAINT medicine_name IF EXISTS",
    "DROP CONSTRAINT ingredient_name IF EXISTS",
    "DROP CONSTRAINT side_effect_name IF EXISTS",
    "DROP CONSTRAINT condition_name IF EXISTS",
    "DROP CONSTRAINT manufacturer_name IF EXISTS",
    "CREATE CONSTRAINT medicine_id IF NOT EXISTS FOR (m:Medicine) REQUIRE m.medicine_id IS UNIQUE",
    "CREATE CONSTRAINT data_source_id IF NOT EXISTS FOR (s:DataSource) REQUIRE s.source_id IS UNIQUE",
    "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:ActiveIngredient) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT side_effect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT manufacturer_name IF NOT EXISTS FOR (mf:Manufacturer) REQUIRE mf.name IS UNIQUE",
    f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS FOR (m:Medicine) ON EACH [m.name, m.composition, m.uses_text, m.side_effects_text]"
]

VECTOR_INDEX_CYPHER = f"""
CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
FOR (m:Medicine) ON m.embedding
OPTIONS {{ indexConfig: {{ `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' }} }}
"""

MERGE_MEDICINE_CYPHER = """
MERGE (m:Medicine {medicine_id: $medicine_id})
SET m.name=$name,
    m.composition=$composition,
    m.manufacturer=$manufacturer,
    m.normalized_name=$normalized_name,
    m.normalized_composition=$normalized_composition,
    m.normalized_manufacturer=$normalized_manufacturer,
    m.ingredients=$ingredients,
    m.normalized_ingredients=$normalized_ingredients,
    m.uses_text=$uses_text,
    m.side_effects_text=$side_effects_text,
    m.image_url=$image_url,
    m.excellent_review_pct=$excellent_review_pct,
    m.average_review_pct=$average_review_pct,
    m.poor_review_pct=$poor_review_pct,
    m.source_id=$source_id,
    m.source_name=$source_name,
    m.source_url=$source_url,
    m.source_date=$source_date,
    m.source_retrieved_at=$source_retrieved_at,
    m.source_license=$source_license,
    m.source_sha256=$source_sha256,
    m.source_row_ids=$source_row_ids,
    m.source_row_numbers=$source_row_numbers,
    m.source_row_count=$source_row_count,
    m.source_record_json=$source_record_json,
    m.source_variants_json=$source_variants_json,
    m.source_conflict_fields=$source_conflict_fields,
    m.jurisdiction=$jurisdiction,
    m.content_type=$content_type,
    m.clinical_authority=$clinical_authority,
    m.provenance_status=$provenance_status,
    m.ingested_at=$ingested_at
"""

MERGE_EMBEDDING_CYPHER = """
MATCH (m:Medicine {medicine_id: $medicine_id})
SET m.embedding=$embedding
"""

MERGE_SOURCE_CYPHER = """
MERGE (s:DataSource {source_id: $source_id})
SET s.name=$source_name, s.url=$source_url, s.source_date=$source_date,
    s.retrieved_at=$source_retrieved_at, s.license=$source_license,
    s.sha256=$source_sha256, s.jurisdiction=$jurisdiction,
    s.content_type=$content_type, s.clinical_authority=$clinical_authority,
    s.provenance_status=$provenance_status, s.ingested_at=$ingested_at
"""

MERGE_SOURCE_REL = """
MATCH (m:Medicine {medicine_id: $medicine_id})
MATCH (s:DataSource {source_id: $source_id})
MERGE (m)-[r:DERIVED_FROM {source_id: $source_id}]->(s)
SET r.source_row_ids=$source_row_ids, r.source_row_numbers=$source_row_numbers,
    r.source_record_count=$source_row_count, r.source_sha256=$source_sha256
"""

MERGE_MANUFACTURER_REL = """
MERGE (mf:Manufacturer {name: $manufacturer})
SET mf.normalized_name=$normalized_manufacturer
MERGE (m:Medicine {medicine_id: $medicine_id})
MERGE (m)-[:MANUFACTURED_BY]->(mf)
"""

MERGE_INGREDIENT_REL = """
MERGE (i:ActiveIngredient {name: $ingredient})
SET i.normalized_name=$normalized_ingredient
MERGE (m:Medicine {medicine_id: $medicine_id})
MERGE (m)-[:CONTAINS_INGREDIENT]->(i)
"""

MERGE_SIDE_EFFECT_REL = """
MERGE (s:SideEffect {name: $side_effect})
SET s.normalized_name=$normalized_side_effect
MERGE (m:Medicine {medicine_id: $medicine_id})
MERGE (m)-[:HAS_SIDE_EFFECT]->(s)
"""

MERGE_CONDITION_REL = """
MERGE (c:Condition {name: $condition})
SET c.normalized_name=$normalized_condition
MERGE (m:Medicine {medicine_id: $medicine_id})
MERGE (m)-[:TREATS]->(c)
"""

def build_embedding_text(row: dict) -> str:
    parts = [row.get('name', ''), row.get('composition', ''), row.get('uses_text', ''), row.get('side_effects_text', ''), row.get('manufacturer', '')]
    return ' | '.join(p for p in parts if p)


def backfill_legacy_medicine_ids(session) -> int:
    """Assign deterministic identities to nodes created by the earlier name-keyed schema."""
    existing = list(session.run("""
        MATCH (m:Medicine)
        WHERE m.medicine_id IS NULL
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
        RETURN elementId(m) AS element_id, m.name AS name,
               m.composition AS composition, head(collect(DISTINCT mf.name)) AS manufacturer
    """))
    updates = []
    for record in existing:
        name = record.get('name') or ''
        composition = record.get('composition') or ''
        manufacturer = record.get('manufacturer') or 'Unknown'
        updates.append({
            'element_id': record['element_id'],
            'medicine_id': stable_medicine_id(name, composition, manufacturer),
            'normalized_name': normalize_key(name),
            'normalized_composition': normalize_key(composition),
            'normalized_manufacturer': normalize_key(manufacturer),
        })
    if updates:
        session.run("""
            UNWIND $rows AS row
            MATCH (m:Medicine) WHERE elementId(m)=row.element_id
            SET m.medicine_id=row.medicine_id,
                m.normalized_name=row.normalized_name,
                m.normalized_composition=row.normalized_composition,
                m.normalized_manufacturer=row.normalized_manufacturer
        """, rows=updates).consume()
    return len(updates)

def ingest(
    csv_path: str,
    limit: int | None = None,
    clear: bool = False,
    no_embeddings: bool = False,
    rebuild_managed_relationships: bool = False,
    manifest_path: str = str(DEFAULT_MANIFEST),
):
    driver = get_driver()
    records, quality_report, _ = prepare_records(csv_path, manifest_path)
    if limit is not None:
        records = records[:limit]

    model = None
    if not no_embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)

    log_event("ingestion_started", source_id=quality_report["source_id"], source_sha256=quality_report["source_sha256"], source_rows=quality_report["source_rows"], requested_limit=limit)
    try:
        with driver.session() as session:
            if clear:
                session.run("MATCH (n) DETACH DELETE n").consume()
            migrated_count = backfill_legacy_medicine_ids(session)
            for query in SCHEMA_QUERIES:
                session.run(query).consume()
            if rebuild_managed_relationships:
                session.run("""
                    MATCH (m:Medicine)-[r]->()
                    WHERE type(r) IN ['CONTAINS_INGREDIENT','HAS_SIDE_EFFECT','TREATS','MANUFACTURED_BY','DERIVED_FROM']
                    DELETE r
                """).consume()
            if not no_embeddings:
                session.run(VECTOR_INDEX_CYPHER).consume()

            source_values = {
                'source_id': records[0]['source_id'] if records else quality_report['source_id'],
                'source_name': records[0]['source_name'] if records else quality_report['source_name'],
                'source_url': records[0]['source_url'] if records else quality_report['source_url'],
                'source_date': records[0]['source_date'] if records else quality_report['source_date'],
                'source_retrieved_at': records[0]['source_retrieved_at'] if records else quality_report['source_retrieved_at'],
                'source_license': records[0]['source_license'] if records else quality_report['license'],
                'source_sha256': quality_report['source_sha256'],
                'jurisdiction': records[0]['jurisdiction'] if records else quality_report['jurisdiction'],
                'content_type': records[0]['content_type'] if records else 'medicine_catalog',
                'clinical_authority': records[0]['clinical_authority'] if records else 'not_assessed',
                'provenance_status': records[0]['provenance_status'] if records else quality_report['provenance_status'],
                'ingested_at': records[0]['ingested_at'] if records else quality_report['generated_at'],
            }
            session.run(MERGE_SOURCE_CYPHER, source_values).consume()

            for row in tqdm(records, total=len(records), desc="Loading medicines"):
                parameters = {
                    **row,
                    'excellent_review_pct': row['excellent_review_pct'] or 0,
                    'average_review_pct': row['average_review_pct'] or 0,
                    'poor_review_pct': row['poor_review_pct'] or 0,
                }
                session.run(MERGE_MEDICINE_CYPHER, parameters).consume()
                session.run(MERGE_MANUFACTURER_REL, {
                    'medicine_id': row['medicine_id'],
                    'manufacturer': row['manufacturer'],
                    'normalized_manufacturer': row['normalized_manufacturer'],
                }).consume()
                session.run(MERGE_SOURCE_REL, {
                    'medicine_id': row['medicine_id'],
                    'source_id': row['source_id'],
                    'source_row_ids': row['source_row_ids'],
                    'source_row_numbers': row['source_row_numbers'],
                    'source_row_count': row['source_row_count'],
                    'source_sha256': row['source_sha256'],
                }).consume()

                for ingredient in parse_active_ingredients(row['composition']):
                    session.run(MERGE_INGREDIENT_REL, {
                        'medicine_id': row['medicine_id'],
                        'ingredient': ingredient,
                        'normalized_ingredient': normalize_key(ingredient),
                    }).consume()
                for side_effect in parse_side_effects(row['side_effects_text']):
                    session.run(MERGE_SIDE_EFFECT_REL, {
                        'medicine_id': row['medicine_id'],
                        'side_effect': side_effect,
                        'normalized_side_effect': normalize_key(side_effect),
                    }).consume()
                for condition in extract_conditions(row['uses_text']):
                    session.run(MERGE_CONDITION_REL, {
                        'medicine_id': row['medicine_id'],
                        'condition': condition,
                        'normalized_condition': normalize_key(condition),
                    }).consume()
                if model is not None:
                    embedding = model.encode(build_embedding_text(row)).tolist()
                    session.run(MERGE_EMBEDDING_CYPHER, {
                        'medicine_id': row['medicine_id'],
                        'embedding': embedding,
                    }).consume()

        log_event("ingestion_completed", source_id=quality_report["source_id"], source_sha256=quality_report["source_sha256"], source_rows=quality_report["source_rows"], products_loaded=len(records), legacy_nodes_migrated=migrated_count, embeddings_loaded=not no_embeddings, rebuild_managed_relationships=rebuild_managed_relationships)
        print("Ingestion complete.")
        print(f"Loaded {len(records)} product identities from {quality_report['source_rows']} source rows.")
        print(f"Source provenance: {quality_report['provenance_status']}; SHA-256: {quality_report['source_sha256']}")
        print(f"Legacy medicine nodes assigned stable IDs: {migrated_count}")
        if no_embeddings:
            print("Embeddings were not generated.")
        else:
            print(f"Embeddings and vector index: {VECTOR_INDEX_NAME}")
    except Exception as exc:
        log_event("ingestion_failed", source_id=quality_report.get("source_id"), source_sha256=quality_report.get("source_sha256"), error_type=type(exc).__name__)
        raise
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest medicine CSV into Neo4j graph")
    parser.add_argument('--csv', default='data/Medicine_Details.csv', help='Path to CSV file')
    parser.add_argument('--limit', type=int, help='Limit rows for debugging')
    parser.add_argument('--clear', action='store_true', help='Clear existing graph data first')
    parser.add_argument('--no-embeddings', action='store_true', help='Create the graph without loading the embedding model')
    parser.add_argument('--rebuild-managed-relationships', action='store_true', help='Recreate catalog-derived relationships from the CSV before ingesting')
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST), help='Dataset provenance manifest JSON')
    args = parser.parse_args()
    ingest(args.csv, args.limit, args.clear, args.no_embeddings, args.rebuild_managed_relationships, args.manifest)

if __name__ == '__main__':
    main()
