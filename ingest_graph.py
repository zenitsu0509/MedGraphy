import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Default embedding model (384-dim)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "medicine_embeddings")


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
    user = os.getenv('NEO4J_USER')
    pwd = os.getenv('NEO4J_PASSWORD')
    if not all([uri, user, pwd]):
        raise ValueError("Missing Neo4j credentials in environment (.env)")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    return driver

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT medicine_name IF NOT EXISTS FOR (m:Medicine) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:ActiveIngredient) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT side_effect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT manufacturer_name IF NOT EXISTS FOR (mf:Manufacturer) REQUIRE mf.name IS UNIQUE"
]

VECTOR_INDEX_CYPHER = f"""
CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
FOR (m:Medicine) ON m.embedding
OPTIONS {{ indexConfig: {{ `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' }} }}
"""

MERGE_MEDICINE_CYPHER = """
MERGE (m:Medicine {name: $name})
SET m.composition=$composition,
    m.uses_text=$uses_text,
    m.side_effects_text=$side_effects_text,
    m.image_url=$image_url,
    m.excellent_review_pct=$excellent_review_pct,
    m.average_review_pct=$average_review_pct,
    m.poor_review_pct=$poor_review_pct,
    m.embedding=$embedding
"""

MERGE_MANUFACTURER_REL = """
MERGE (mf:Manufacturer {name: $manufacturer})
MERGE (m:Medicine {name: $medicine_name})
MERGE (m)-[:MANUFACTURED_BY]->(mf)
"""

MERGE_INGREDIENT_REL = """
MERGE (i:ActiveIngredient {name: $ingredient})
MERGE (m:Medicine {name: $medicine_name})
MERGE (m)-[:CONTAINS_INGREDIENT]->(i)
"""

MERGE_SIDE_EFFECT_REL = """
MERGE (s:SideEffect {name: $side_effect})
MERGE (m:Medicine {name: $medicine_name})
MERGE (m)-[:HAS_SIDE_EFFECT]->(s)
"""

MERGE_CONDITION_REL = """
MERGE (c:Condition {name: $condition})
MERGE (m:Medicine {name: $medicine_name})
MERGE (m)-[:TREATS]->(c)
"""

CREATE_SHARED_INGREDIENT_REL = """
MATCH (i:ActiveIngredient)<-[:CONTAINS_INGREDIENT]-(m1:Medicine), (i)<-[:CONTAINS_INGREDIENT]-(m2:Medicine)
WHERE id(m1) < id(m2)
MERGE (m1)-[:INTERACTS_WITH {basis:'shared_ingredient', ingredient: i.name}]->(m2)
"""

def build_embedding_text(row: pd.Series) -> str:
    parts = [str(row.get('Medicine Name','')), str(row.get('Composition','')), str(row.get('Uses','')), str(row.get('Side_effects','')), str(row.get('Manufacturer',''))]
    return ' | '.join(p for p in parts if p and p != 'nan')

def ingest(csv_path: str, limit: int | None = None, clear: bool = False):
    driver = get_driver()
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    model = SentenceTransformer(EMBEDDING_MODEL)

    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
        for q in SCHEMA_QUERIES:
            session.run(q)
        session.run(VECTOR_INDEX_CYPHER)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading medicines"):
            name = row.get('Medicine Name')
            composition = row.get('Composition','')
            uses_text = row.get('Uses','')
            side_effects_raw = row.get('Side_effects','')
            image_url = row.get('Image URL','')
            manufacturer = row.get('Manufacturer','Unknown')
            excellent = int(row.get('Excellent Review %',0)) if not pd.isna(row.get('Excellent Review %')) else 0
            average = int(row.get('Average Review %',0)) if not pd.isna(row.get('Average Review %')) else 0
            poor = int(row.get('Poor Review %',0)) if not pd.isna(row.get('Poor Review %')) else 0

            embedding_text = build_embedding_text(row)
            embedding = model.encode(embedding_text).tolist()

            session.run(MERGE_MEDICINE_CYPHER, {
                'name': name,
                'composition': composition,
                'uses_text': uses_text,
                'side_effects_text': side_effects_raw,
                'image_url': image_url,
                'excellent_review_pct': excellent,
                'average_review_pct': average,
                'poor_review_pct': poor,
                'embedding': embedding
            })

            session.run(MERGE_MANUFACTURER_REL, {'manufacturer': manufacturer, 'medicine_name': name})

            # Active Ingredients
            for ing in parse_active_ingredients(composition):
                session.run(MERGE_INGREDIENT_REL, {'ingredient': ing, 'medicine_name': name})

            # Side Effects
            for se in parse_side_effects(side_effects_raw):
                session.run(MERGE_SIDE_EFFECT_REL, {'side_effect': se, 'medicine_name': name})

            # Conditions
            for cond in extract_conditions(uses_text):
                session.run(MERGE_CONDITION_REL, {'condition': cond, 'medicine_name': name})

        # Interaction relationships (shared ingredient)
        session.run(CREATE_SHARED_INGREDIENT_REL)

    driver.close()
    print("Ingestion complete.")
    print(f"Loaded {len(df)} medicines. Vector index: {VECTOR_INDEX_NAME}")


def main():
    parser = argparse.ArgumentParser(description="Ingest medicine CSV into Neo4j graph")
    parser.add_argument('--csv', default='data/Medicine_Details.csv', help='Path to CSV file')
    parser.add_argument('--limit', type=int, help='Limit rows for debugging')
    parser.add_argument('--clear', action='store_true', help='Clear existing graph data first')
    args = parser.parse_args()
    ingest(args.csv, args.limit, args.clear)

if __name__ == '__main__':
    main()
