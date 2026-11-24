import os
import re
import time
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from typing import Iterable
from data_pipeline import normalize_key
from graph_db import Neo4jConnection
from sentence_transformers import SentenceTransformer
from monitoring import hash_query, log_event, source_freshness

# --- CONFIGURATION ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "medicine_embeddings")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME", "medicine_catalog_fulltext")
if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", FULLTEXT_INDEX_NAME):
    raise ValueError("FULLTEXT_INDEX_NAME must be a simple Neo4j identifier")


def reciprocal_rank_fusion(
    rankings: Iterable[tuple[str, Iterable[dict], float]],
    rank_constant: int = 60,
    limit: int | None = None,
) -> list[dict]:
    """Fuse ordered result lists by ID using weighted Reciprocal Rank Fusion."""
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    fused: dict[str, dict] = {}
    for source, candidates, weight in rankings:
        if weight < 0:
            raise ValueError("source weights must be non-negative")
        for rank, candidate in enumerate(candidates, start=1):
            identity = candidate.get("medicine_id") or candidate.get("name")
            if not identity:
                continue
            item = fused.setdefault(identity, {
                "medicine_id": candidate.get("medicine_id"),
                "name": candidate.get("name"),
                "rrf_score": 0.0,
                "sources": [],
            })
            item["rrf_score"] += weight / (rank_constant + rank)
            if source not in item["sources"]:
                item["sources"].append(source)
    ranked = sorted(fused.values(), key=lambda item: (-item["rrf_score"], (item["name"] or "").casefold()))
    return ranked[:limit] if limit is not None else ranked


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

    def vector_similarity_search(self, query: str, limit: int = 5) -> list[dict]:
        """Uses embeddings to find semantically similar medicines."""
        limit = max(1, min(int(limit), 50))
        query_embedding = self.get_embedding(query)
        cypher_query = """
            CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
            YIELD node AS medicine, score
            RETURN medicine.medicine_id AS medicine_id, medicine.name AS name, score
            ORDER BY score DESC
        """
        result = self.db.query(
            cypher_query,
            parameters={"index_name": VECTOR_INDEX_NAME, "limit": limit, "embedding": query_embedding},
            db="neo4j",
        )
        log_event("retrieval", route="vector", hit_count=len(result or []), outcome="hit" if result else "miss")
        return result

    def fulltext_search(self, query: str, limit: int = 10) -> list[dict]:
        """Run Lucene full-text retrieval over product names and catalog text."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for", "from",
            "how", "i", "in", "is", "it", "lists", "listed", "me", "medicine", "medgraphy",
            "of", "on", "or", "product", "should", "the", "this", "to", "use", "what", "which",
            "with", "would",
        }
        tokens = []
        for token in re.findall(r"[a-z0-9]+", query.lower()):
            if len(token) < 2 or token in stopwords or token in tokens:
                continue
            tokens.append(token)
        if not tokens:
            return []
        # Quote each token so user-provided Lucene operators cannot change query semantics.
        search_text = " OR ".join(f'"{token}"' for token in tokens[:16])
        try:
            return self.db.query("""
                CALL db.index.fulltext.queryNodes($index_name, $search_text, {limit:$limit})
                YIELD node AS medicine, score
                RETURN medicine.medicine_id AS medicine_id, medicine.name AS name, score
                ORDER BY score DESC
            """, {"index_name": FULLTEXT_INDEX_NAME, "search_text": search_text, "limit": limit}, db="neo4j") or []
        except Exception as exc:
            # Keep old databases usable until the explicit index migration has run.
            log_event("retrieval_fulltext_unavailable", index_name=FULLTEXT_INDEX_NAME, error_type=type(exc).__name__)
            return []

    def condition_candidates(self, condition: str, limit: int = 10) -> list[dict]:
        """Return condition/use matches ordered by lexical specificity, not review scores."""
        if not condition:
            return []
        limit = max(1, min(int(limit), 50))
        rows = self.db.query("""
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
                   match_priority, m.uses_text AS uses_text
            ORDER BY match_priority, size(coalesce(m.uses_text,'')), name
            LIMIT $limit
        """, {"cond": condition, "limit": limit}, db="neo4j")
        return rows or []

    def _exact_named_medicine(self, user_query: str) -> dict | None:
        """Find a catalog product mention using its normalized display name."""
        rows = self.db.query("""
            MATCH (m:Medicine)
            WHERE ($normalized_query <> '' AND m.normalized_name IS NOT NULL
                   AND $normalized_query CONTAINS m.normalized_name)
               OR (m.normalized_name IS NULL AND toLower($query) CONTAINS toLower(m.name))
            RETURN m.name AS name, m.medicine_id AS medicine_id
            ORDER BY size(coalesce(m.normalized_name, toLower(m.name))) DESC, m.name
            LIMIT 1
        """, {"query": user_query.strip(), "normalized_query": normalize_key(user_query)}, db="neo4j") or []
        return dict(rows[0]) if rows else None

    
    def _best_medicine_for_condition(self, condition: str) -> dict | None:
        """Return the most lexically specific catalog match without review-score ranking."""
        rows = self.condition_candidates(condition, limit=1)
        return dict(rows[0]) if rows else None

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
        list_match = re.search(r"\blists?\s+", q)
        if list_match:
            tail = q[list_match.end():]
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

    def retrieve_context_for_rag(
        self,
        user_query: str,
        strategy: str = "hybrid",
        preferred_medicine_id: str | None = None,
    ) -> dict | None:
        """Retrieve one catalog record using vector-only or lexical/graph/vector RRF."""
        if strategy not in {"hybrid", "vector-only"}:
            raise ValueError("strategy must be 'hybrid' or 'vector-only'")
        started = time.monotonic()
        top_medicine_name: str | None = None
        top_medicine_id: str | None = None
        retrieval_sources: list[str] = []
        retrieval_score: float | None = None
        candidates_by_id: dict[str, dict[str, Any]] = {}

        if strategy == "vector-only":
            vector_rows = self.vector_similarity_search(user_query, limit=1)
            ranked = reciprocal_rank_fusion([("vector", vector_rows, 1.0)], rank_constant=0, limit=1)
            candidates_by_id = {item.get("medicine_id") or item.get("name"): item for item in ranked}
        else:
            # Exact named products should win over semantically similar records.
            named = self._exact_named_medicine(user_query)
            if named:
                top_medicine_name = named["name"]
                top_medicine_id = named.get("medicine_id")
                retrieval_sources = ["exact_name"]
                retrieval_score = 1.0

            # Keep follow-up questions anchored to the active medicine from this chat.
            # An exact product name in the new question always takes precedence.
            if not top_medicine_name and preferred_medicine_id:
                preferred_rows = self.db.query("""
                    MATCH (m:Medicine {medicine_id:$medicine_id})
                    RETURN m.name AS name, m.medicine_id AS medicine_id
                    LIMIT 1
                """, {"medicine_id": preferred_medicine_id}, db="neo4j") or []
                if preferred_rows:
                    top_medicine_name = preferred_rows[0]["name"]
                    top_medicine_id = preferred_rows[0]["medicine_id"]
                    retrieval_sources = ["conversation_context"]
                    retrieval_score = 1.0

            if not top_medicine_name:
                rankings = []
                condition = self._extract_condition_from_query(user_query)
                if condition:
                    rankings.append(("condition_graph", self.condition_candidates(condition, limit=20), 2.0))
                rankings.append(("fulltext", self.fulltext_search(user_query, limit=20), 1.0))
                rankings.append(("vector", self.vector_similarity_search(user_query, limit=20), 1.0))
                ranked = reciprocal_rank_fusion(rankings, rank_constant=60, limit=20)
                candidates_by_id = {item.get("medicine_id") or item.get("name"): item for item in ranked}
            if top_medicine_name:
                candidates_by_id = {
                    top_medicine_id or top_medicine_name: {
                        "medicine_id": top_medicine_id, "name": top_medicine_name,
                        "rrf_score": 1.0, "sources": retrieval_sources,
                    }
                }

        if not top_medicine_name and candidates_by_id:
            winner = sorted(
                candidates_by_id.values(),
                key=lambda item: (-item["rrf_score"], (item["name"] or "").casefold()),
            )[0]
            top_medicine_name = winner["name"]
            top_medicine_id = winner.get("medicine_id")
            retrieval_sources = winner["sources"]
            retrieval_score = winner["rrf_score"]
        if not top_medicine_name:
            log_event("retrieval", route=f"rag_{strategy}", hit_count=0, outcome="miss", query_hash=hash_query(user_query))
            return None

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
        context["retrieval_strategy"] = strategy
        context["retrieval_sources"] = retrieval_sources
        context["retrieval_score"] = retrieval_score
        log_event("retrieval_ranked", route=f"rag_{strategy}", candidate_count=len(candidates_by_id), selected_sources=retrieval_sources)
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
