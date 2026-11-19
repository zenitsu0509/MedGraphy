import os
import re
import time
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from graph_db import Neo4jConnection
from sentence_transformers import SentenceTransformer
from monitoring import hash_query, log_event, source_freshness

# --- CONFIGURATION ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "medicine_embeddings")


@lru_cache(maxsize=1)
def get_embedding_model():
    """Initializes and returns the SentenceTransformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


class GraphQueryEngine:
    """Handles queries to the Neo4j database."""

    def __init__(self, db=None, model=None):
        self.db = db or Neo4jConnection()
        self.model = model or get_embedding_model()
        print("Embedding model loaded")

    # ----------------------------
    # Basic helpers
    # ----------------------------
    def get_embedding(self, text: str) -> list[float]:
        """Generates an embedding for a given text."""
        return self.model.encode(text).tolist()

    # ----------------------------
    # Core queries
    # ----------------------------
    def direct_lookup(self, medicine_name: str) -> list[dict]:
        """Finds uses and side effects for a specific medicine."""
        query = """
            MATCH (m:Medicine {name: $med_name})
            OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            RETURN m.name AS medicine,
                   collect(DISTINCT c.name) AS uses,
                   collect(DISTINCT s.name) AS side_effects
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result

    def reverse_lookup(self, condition: str) -> list[str]:
        """Finds medicines that treat a specific condition (case-insensitive).
        Falls back to searching uses_text if Condition nodes are missing.
        """
        query = """
            WITH $cond_name AS q
            CALL (q) {
              MATCH (m:Medicine)-[:TREATS]->(c:Condition)
              WHERE toLower(c.name) = toLower(q)
              RETURN DISTINCT m.name AS medicine, 0 AS priority
              UNION
              MATCH (m:Medicine)-[:TREATS]->(c:Condition)
              WHERE toLower(c.name) CONTAINS toLower(q)
              RETURN DISTINCT m.name AS medicine, 1 AS priority
              UNION
              MATCH (m:Medicine)
              WHERE m.uses_text IS NOT NULL AND toLower(m.uses_text) CONTAINS toLower(q)
              RETURN DISTINCT m.name AS medicine, 2 AS priority
            }
            WITH medicine, min(priority) AS priority
            RETURN medicine ORDER BY priority, medicine
            LIMIT 25
        """
        result = self.db.query(query, parameters={"cond_name": condition}, db="neo4j")
        medicines = [record["medicine"] for record in result] if result else []
        log_event("retrieval", route="condition", hit_count=len(medicines), outcome="hit" if medicines else "miss")
        return medicines

    def suggest_condition_query(self, condition: str) -> str | None:
        """Suggest a one-token spelling correction using condition and catalog vocabulary."""
        query_tokens = re.findall(r"[a-z]+", condition.lower())
        if not query_tokens:
            return None
        vocabulary = getattr(self, "_condition_vocabulary", None)
        if vocabulary is None:
            vocabulary = set()
            for row in self.db.query("""
                MATCH (m:Medicine)
                WHERE m.uses_text IS NOT NULL
                RETURN DISTINCT m.uses_text AS value
            """, db="neo4j") or []:
                vocabulary.update(re.findall(r"[a-z]+", (row.get("value") or "").lower()))
            for row in self.db.query("MATCH (c:Condition) RETURN DISTINCT c.name AS value", db="neo4j") or []:
                vocabulary.update(re.findall(r"[a-z]+", (row.get("value") or "").lower()))
            self._condition_vocabulary = vocabulary

        ignored = {"and", "for", "from", "of", "the", "with", "due", "treatment", "prevention"}
        corrected = list(query_tokens)
        changes = 0
        for index, token in enumerate(query_tokens):
            if len(token) < 4 or token in ignored or token in vocabulary:
                continue
            matches = get_close_matches(token, vocabulary, n=2, cutoff=0.80)
            if not matches:
                continue
            best = matches[0]
            best_score = SequenceMatcher(None, token, best).ratio()
            second_score = SequenceMatcher(None, token, matches[1]).ratio() if len(matches) > 1 else 0
            if best_score >= 0.80 and best_score - second_score >= 0.04:
                corrected[index] = best
                changes += 1
        if changes == 1:
            return " ".join(corrected)
        return None

    def shared_ingredient_products(self, medicine_name: str) -> list[dict]:
        """Returns catalog products sharing ingredients; this is not interaction evidence."""
        query = """
            MATCH (m1:Medicine {name: $med_name})-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            MATCH (m2:Medicine)-[:CONTAINS_INGREDIENT]->(i)
            WHERE m1 <> m2
            RETURN m2.medicine_id AS medicine_id, m2.name AS other_medicine,
                   i.name AS shared_ingredient
            LIMIT 10
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result

    def check_interactions(self, medicine_name: str) -> list[dict]:
        """Compatibility alias for legacy callers; results are ingredient overlaps only."""
        return self.shared_ingredient_products(medicine_name)

    def known_interactions(self, medicine_name: str) -> list[dict]:
        query = """
        MATCH (m:Medicine {name:$name})-[r]-(o:Medicine)
        WHERE type(r) = 'HAS_KNOWN_INTERACTION'
        RETURN o.medicine_id AS medicine_id, o.name AS other_medicine,
               properties(r) AS evidence
        LIMIT 25
        """
        rows = self.db.query(query, {"name": medicine_name}, db="neo4j") or []
        return [{**dict(row), **(row.get("evidence") or {})} for row in rows]

    def vector_similarity_search(self, query: str) -> list[dict]:
        """Uses embeddings to find semantically similar medicines."""
        query_embedding = self.get_embedding(query)
        cypher_query = """
            CALL db.index.vector.queryNodes($index_name, 5, $embedding)
            YIELD node AS medicine, score
            RETURN medicine.medicine_id AS medicine_id, medicine.name AS name, score
        """
        result = self.db.query(
            cypher_query,
            parameters={"index_name": VECTOR_INDEX_NAME, "embedding": query_embedding},
            db="neo4j",
        )
        log_event("retrieval", route="vector", hit_count=len(result or []), outcome="hit" if result else "miss")
        return result

    
    def _best_medicine_for_condition(self, condition: str) -> dict | None:
        """Prefer the most specific catalog match; use review scores only as a tie-breaker."""
        if not condition:
            return None
        cypher = """
        WITH $cond AS q
        CALL (q) {
          MATCH (m:Medicine)-[:TREATS]->(c:Condition)
          WHERE toLower(c.name) = toLower(q)
          RETURN m, 0 AS match_priority
          UNION
          MATCH (m:Medicine)-[:TREATS]->(c:Condition)
          WHERE toLower(c.name) CONTAINS toLower(q) AND toLower(c.name) <> toLower(q)
          RETURN m, 1 AS match_priority
          UNION
          MATCH (m:Medicine)
          WHERE m.uses_text IS NOT NULL AND toLower(m.uses_text) CONTAINS toLower(q)
          RETURN m, 2 AS match_priority
        }
        WITH m, min(match_priority) AS match_priority
        RETURN m.name AS name, m.medicine_id AS medicine_id,
               coalesce(m.excellent_review_pct,0) AS excellent,
               coalesce(m.average_review_pct,0) AS average
        ORDER BY match_priority, excellent DESC, average DESC, name
        LIMIT 1
        """
        res = self.db.query(cypher, {"cond": condition}, db="neo4j")
        if res and len(res) > 0:
            try:
                return {"name": res[0]["name"], "medicine_id": res[0]["medicine_id"]}
            except Exception:
                rec = res[0]
                if hasattr(rec, "data"):
                    return rec.data()
                if hasattr(rec, "keys"):
                    return {"name": rec["name"], "medicine_id": rec["medicine_id"]}
        return None

    def _extract_condition_from_query(self, user_query: str) -> str | None:
        """Very light heuristic to extract a condition phrase like 'fever' from queries.
        Examples: 'medicine for fever', 'drug for high fever', 'fever medicine'.
        """
        if not user_query:
            return None
        q = user_query.strip().lower()
        # Rule 1: look for ' for <cond>' pattern
        if " for " in q:
            tail = q.split(" for ", 1)[1]
            for sep in ["?", ".", ",", ";", ":"]:
                tail = tail.split(sep)[0]
            cond = tail.strip()
            cond = re.sub(r"\s+(?:as\s+(?:a\s+)?use|as\s+a\s+listed\s+use|in\s+the\s+catalog)$", "", cond)
            for lead in ["a ", "an ", "the "]:
                if cond.startswith(lead):
                    cond = cond[len(lead):]
            cond = cond.strip()
            if cond:
                return cond
        if " lists " in q:
            tail = q.split(" lists ", 1)[1]
            tail = re.split(r"\s+as\s+(?:a\s+)?use\b|[?.,;:]", tail, maxsplit=1)[0]
            cond = tail.strip()
            if cond.startswith("a "):
                cond = cond[2:]
            if cond:
                return cond
        # Rule 2: '<cond> medicine' pattern
        if q.endswith(" medicine"):
            cond = q.rsplit(" ", 1)[0].strip()
            return cond or None
        # Rule 3: common single-word conditions
        common = {"fever", "cold", "cough", "pain", "migraine", "diarrhea", "diarrhoea"}
        tokens = [t for t in q.replace("?", " ").split() if t.isalpha()]
        for t in tokens:
            if t in common:
                return t
        return None

    def retrieve_context_for_rag(self, user_query: str) -> dict | None:
        """Retrieves the context from the graph for the RAG pipeline.
        Strategy: Try condition-first (e.g., 'medicine for fever') -> choose a best medicine.
        Fall back to vector retrieval if condition-based selection fails.
        """
        started = time.monotonic()
        top_medicine_name: str | None = None
        top_medicine_id: str | None = None

        # A product explicitly named in the prompt is a stronger anchor than
        # semantic similarity, which can otherwise return a related product.
        named = self.db.query("""
            MATCH (m:Medicine)
            WHERE toLower($query) CONTAINS toLower(m.name)
            RETURN m.name AS name, m.medicine_id AS medicine_id
            ORDER BY size(m.name) DESC, m.name
            LIMIT 1
        """, {"query": user_query.strip()}, db="neo4j")
        if named:
            top_medicine_name = named[0]["name"]
            top_medicine_id = named[0].get("medicine_id")

        # A) Condition-first heuristic
        cond = self._extract_condition_from_query(user_query)
        if cond and not top_medicine_name:
            best = self._best_medicine_for_condition(cond)
            if best:
                top_medicine_name = best["name"]
                top_medicine_id = best.get("medicine_id")

        # B) Vector fallback if no clear condition-based anchor
        if not top_medicine_name:
            query_embedding = self.get_embedding(user_query)
            retrieval_query = """
                CALL db.index.vector.queryNodes($index_name, 1, $embedding)
                YIELD node AS medicine
                RETURN medicine.name AS med_name, medicine.medicine_id AS medicine_id
            """
            retrieval_result = self.db.query(
                retrieval_query,
                parameters={"index_name": VECTOR_INDEX_NAME, "embedding": query_embedding},
                db="neo4j",
            )
            if not retrieval_result:
                log_event("retrieval", route="rag_vector_fallback", hit_count=0, outcome="miss", query_hash=hash_query(user_query))
                return None
            top_medicine_name = retrieval_result[0]["med_name"]
            top_medicine_id = retrieval_result[0].get("medicine_id")

        # 2. AUGMENT: Fetch its full context from the graph
        context_query = """
            MATCH (m:Medicine)
            WHERE ($medicine_id IS NOT NULL AND m.medicine_id=$medicine_id)
               OR ($medicine_id IS NULL AND m.name=$med_name)
            OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
            RETURN m.name AS medicine,
                   m.composition AS composition,
                   m.uses_text AS uses_text,
                   m.side_effects_text AS side_effects_text,
                   m.image_url AS image_url,
                   m.excellent_review_pct AS excellent_review_pct,
                   m.average_review_pct AS average_review_pct,
                   m.poor_review_pct AS poor_review_pct,
                   m.medicine_id AS medicine_id,
                   properties(m) AS medicine_props,
                   mf.name AS manufacturer,
                   collect(DISTINCT c.name) AS uses,
                   collect(DISTINCT s.name) AS side_effects,
                   collect(DISTINCT i.name) AS ingredients
        """
        context_result = self.db.query(
            context_query, parameters={"med_name": top_medicine_name, "medicine_id": top_medicine_id}, db="neo4j"
        )
        if not context_result:
            log_event("retrieval", route="rag", hit_count=0, outcome="miss", elapsed_ms=round((time.monotonic()-started)*1000))
            return None
        context = dict(context_result[0])
        medicine_props = context.pop("medicine_props", {})
        for key in ("source_id", "source_name", "source_url", "source_date", "source_retrieved_at", "source_license", "source_sha256", "jurisdiction", "clinical_authority", "provenance_status"):
            context[key if key != "source_retrieved_at" else "retrieved_at"] = medicine_props.get(key)
        log_event("retrieval", route="rag", hit_count=1, outcome="hit", elapsed_ms=round((time.monotonic()-started)*1000), query_hash=hash_query(user_query), source_freshness=source_freshness(context.get("retrieved_at")))
        context["source_ref"] = "[S1]"
        return {
            "medicine_found": top_medicine_name,
            "context": context,
        }

    # ----------------------------
    # Utilities for UI
    # ----------------------------
    def get_graph_for_visualization(self, medicine_name: str) -> list:
        """Fetches a subgraph for a given medicine for visualization."""
        query = """
            MATCH path = (m:Medicine {name: $med_name})-[r]-(n)
            RETURN m, r, n
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result if result else []

    def get_medicine_with_image(self, name: str, medicine_id: str | None = None):
        query = """
        MATCH (m:Medicine)
        WHERE ($medicine_id IS NOT NULL AND m.medicine_id=$medicine_id)
           OR ($medicine_id IS NULL AND m.name=$name)
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
        RETURN m.name as name, m.image_url AS image_url, m.composition AS composition,
               m.uses_text AS uses_text, m.side_effects_text AS side_effects_text,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT s.name) AS side_effects,
               collect(DISTINCT i.name) AS ingredients,
               mf.name AS manufacturer,
               m.excellent_review_pct AS excellent_review_pct,
               m.average_review_pct AS average_review_pct,
               m.poor_review_pct AS poor_review_pct,
               m.medicine_id AS medicine_id,
               properties(m) AS medicine_props
        """
        res = self.db.query(query, {"name": name, "medicine_id": medicine_id}, db="neo4j")
        if not res:
            return None
        card = dict(res[0])
        medicine_props = card.pop("medicine_props", {})
        card.update({
            "source_id": medicine_props.get("source_id"),
            "source_name": medicine_props.get("source_name"),
            "source_url": medicine_props.get("source_url"),
            "source_date": medicine_props.get("source_date"),
            "retrieved_at": medicine_props.get("source_retrieved_at"),
            "license": medicine_props.get("source_license"),
            "jurisdiction": medicine_props.get("jurisdiction"),
            "clinical_authority": medicine_props.get("clinical_authority"),
            "provenance_status": medicine_props.get("provenance_status"),
            "source_conflict_fields": medicine_props.get("source_conflict_fields", []),
        })
        return card

    def symptom_to_medicines(self, symptoms: list[str], limit: int = 10):
        # Leverage side effect nodes to map symptoms -> medicines
        query = """
        UNWIND $symptoms AS sym
        MATCH (s:SideEffect)
        WHERE toLower(s.name) CONTAINS toLower(sym)
        MATCH (m:Medicine)-[:HAS_SIDE_EFFECT]->(s)
        RETURN s.name AS matched_symptom, collect(DISTINCT m.name) AS medicines
        LIMIT $limit
        """
        return self.db.query(query, {"symptoms": symptoms, "limit": limit}, db="neo4j")

    def justify_prescription(self, medicines: list[str]):
        # Return structured data for each medicine for LLM justification step
        query = """
        UNWIND $medicines AS med
        MATCH (m:Medicine {name: med})
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
        RETURN m.name AS medicine,
               m.composition AS composition,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT i.name) AS ingredients,
               collect(DISTINCT s.name) AS side_effects
        """
        return self.db.query(query, {"medicines": medicines}, db="neo4j")

    def interaction_conflicts(self, medicine: str):
        """Compatibility alias for evidence-backed interactions only."""
        return self.known_interactions(medicine)
