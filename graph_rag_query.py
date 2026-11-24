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
MIN_VECTOR_SIMILARITY = float(os.getenv("RAG_MIN_VECTOR_SIMILARITY", "0.60"))
FUZZY_PRODUCT_SIMILARITY = float(os.getenv("RAG_FUZZY_PRODUCT_SIMILARITY", "0.82"))
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
            item.setdefault("source_ranks", {})[source] = rank
            if candidate.get("score") is not None and source == "vector":
                item["vector_similarity"] = float(candidate["score"])
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
            "catalog", "catalogue", "entries", "of", "on", "or", "product", "products", "should", "the", "this", "to", "use", "what", "which",
            "list", "lists", "medicine", "medicines", "contains", "containing", "ingredient", "ingredients", "mention", "mentions",
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

    @staticmethod
    def _is_advice_request(query: str) -> bool:
        return bool(re.search(
            r"\b(?:best\s+(?:medicine|medication|drug)|what\s+should\s+i\s+take|"
            r"which\s+(?:medicine|medication|drug)\s+should\s+i\s+take|"
            r"(?:what|which)\s+(?:medicine|medication|drug)\s+for(?:\s+(?:my|this))?\b|"
            r"recommend\s+(?:a\s+)?(?:medicine|medication|drug)|"
            r"what\s+(?:medicine|medication|drug)\s+should\s+i\s+use|"
            r"what\s+(?:medicine|medication|drug)\s+(?:can\s+i\s+take|for\s+my|for\s+this))\b",
            query, re.IGNORECASE,
        ) or re.search(
            r"\b(?:hard\s+to\s+wake|won't\s+wake|trouble\s+breathing|cannot\s+breathe|can't\s+breathe|"
            r"short\s+of\s+breath|lips?\s+(?:turning\s+)?blue|trouble\s+speaking|"
            r"severe\s+chest\s+pain|swallowed\s+(?:a\s+)?(?:cleaning|chemical|poison)|"
            r"too\s+many\s+(?:tablets|pills)|overdose|bleeding\s+heavily|"
            r"sudden\s+weakness\s+on\s+one\s+side|face\s+is\s+swelling)",
            query, re.IGNORECASE,
        ))

    @staticmethod
    def _is_named_product_lookup(query: str) -> bool:
        return bool(re.search(
            r"\b(?:composition|ingredients?|side\s+effects?|manufacturer|details|information)\b"
            r".{0,50}\b(?:for|of|about)\s+[a-z0-9]",
            query, re.IGNORECASE,
        ))

    @staticmethod
    def _is_catalog_field_search(query: str) -> bool:
        return bool(re.search(
            r"\b(?:composition|ingredients?|side\s+effects?|manufacturer|made\s+by|from)\b",
            query, re.IGNORECASE,
        )) and not GraphQueryEngine._is_named_product_lookup(query)

    @staticmethod
    def _manufacturer_query_term(query: str) -> str | None:
        match = re.search(r"\b(?:made\s+by|from|manufacturer\s+(?:is|named)?|company\s+) *([a-z0-9][a-z0-9 &.,'-]{2,70})", query, re.I)
        if not match:
            return None
        maker = re.split(r"\b(?:are|for|that|which|and|with|listed)\b|[?.!,;]", match.group(1), maxsplit=1, flags=re.I)[0].strip()
        if len(maker) < 3:
            return None
        return maker

    def _manufacturer_candidates(self, query: str, limit: int = 20) -> list[dict]:
        maker = self._manufacturer_query_term(query)
        if not maker:
            return []
        return self.db.query("""
            MATCH (m:Medicine)-[:MANUFACTURED_BY]->(mf:Manufacturer)
            WHERE toLower(mf.name) CONTAINS toLower($maker)
            RETURN m.medicine_id AS medicine_id, m.name AS name, mf.name AS manufacturer
            ORDER BY m.name LIMIT $limit
        """, {"maker": maker, "limit": limit}, db="neo4j") or []

    def _condition_manufacturer_candidates(self, condition: str, query: str, limit: int = 50) -> list[dict]:
        maker = self._manufacturer_query_term(query)
        if not maker or not condition:
            return []
        return self.db.query("""
            MATCH (m:Medicine)-[:MANUFACTURED_BY]->(mf:Manufacturer)
            WHERE toLower(mf.name) CONTAINS toLower($maker)
              AND m.uses_text IS NOT NULL AND toLower(m.uses_text) CONTAINS toLower($condition)
            RETURN DISTINCT m.medicine_id AS medicine_id, m.name AS name,
                   m.uses_text AS uses_text, mf.name AS manufacturer
            ORDER BY m.name LIMIT $limit
        """, {"maker": maker, "condition": condition, "limit": limit}, db="neo4j") or []

    def _catalog_field_candidates(self, query: str, limit: int = 20) -> list[dict]:
        q = normalize_key(query)
        if re.search(r"\b(?:side\s+effects?|adverse\s+effects?)\b", query, re.I):
            field = "side_effect"
            ignored = {"side", "effect", "effects", "adverse", "which", "catalog", "catalogue", "products", "medicines", "list", "lists", "mention", "mentions", "their", "among", "the", "as", "a", "an", "what", "are", "is", "for", "in", "that", "have"}
        elif re.search(r"\b(?:composition|ingredients?|contain|contains)\b", query, re.I):
            field = "ingredient"
            ignored = {"composition", "ingredient", "ingredients", "contain", "contains", "which", "catalog", "catalogue", "products", "medicines", "list", "lists", "as", "a", "an", "what", "are", "is", "for", "in", "their", "the"}
        else:
            return []
        terms = [token for token in q.split() if token not in ignored and not token.isdigit()]
        if not terms:
            return []
        # Prefer the most distinctive final field value, e.g. "drowsiness" or "paracetamol".
        term = terms[-1]
        if field == "ingredient" and term == "acetaminophen":
            term = "paracetamol"
        relation, label = ("HAS_SIDE_EFFECT", "SideEffect") if field == "side_effect" else ("CONTAINS_INGREDIENT", "ActiveIngredient")
        return self.db.query(f"""
            MATCH (m:Medicine)-[:{relation}]->(x:{label})
            WHERE x.normalized_name CONTAINS $term
            RETURN m.medicine_id AS medicine_id, m.name AS name, x.name AS matched_field
            ORDER BY m.name LIMIT $limit
        """, {"term": term, "limit": limit}, db="neo4j") or []

    @staticmethod
    def _product_query_text(query: str) -> str:
        text = normalize_key(query)
        prefixes = (
            "what composition is listed for ", "which ingredients are listed for ",
            "give the catalog composition for ", "give catalog composition for ",
            "what side effects are listed for ", "which ingredients are in ",
            "show details for ", "details for ", "information about ",
        )
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        text = re.sub(r"\b(?:please|in the catalog|in catalog|listed|details|composition|ingredients|side effects|manufacturer|what|which|give|show|for|the|a|an|is|are|of|about)\b", " ", text)
        return " ".join(text.split())

    def _fuzzy_product_match(self, query: str, candidates: list[dict] | None = None) -> dict | None:
        product_text = self._product_query_text(query)
        if not product_text:
            return None
        if candidates is None:
            candidates = self.fulltext_search(product_text, limit=50)
        query_tokens = set(product_text.split())
        ranked = []
        for candidate in candidates:
            name = normalize_key(candidate.get("name", ""))
            if not name:
                continue
            name_tokens = set(name.split())
            token_coverage = len(query_tokens & name_tokens) / max(1, len(query_tokens))
            ratio = SequenceMatcher(None, product_text, name).ratio()
            similarity = max(ratio, token_coverage)
            if similarity >= FUZZY_PRODUCT_SIMILARITY and token_coverage >= 0.66:
                ranked.append((similarity, candidate))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], (item[1].get("name") or "").casefold()))
        return {**dict(ranked[0][1]), "name_similarity": round(ranked[0][0], 4)}

    
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
        paraphrase_aliases = {
            "heartburn": "acid reflux", "indigestion": "acid reflux",
            "reflux after meals": "acid reflux", "burning sensation after meals": "acid reflux",
            "fungal disease": "fungal infections", "fungus infection": "fungal infections",
            "bacterial infection": "bacterial infections", "bacterial disease": "bacterial infections",
        }
        for phrase, canonical in paraphrase_aliases.items():
            if phrase in q:
                return canonical
        # Rule 1: look for ' for <cond>' pattern
        if " for " in q and re.search(r"\b(?:medicine|medication|drug|catalog|catalogue|product|products|tablet|capsule|listed|lists?)\b", q):
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
        """Retrieve one record with explicit baselines and conservative abstention."""
        strategies = {"hybrid", "hybrid-no-exact", "vector-only", "fulltext-only", "condition-only", "exact-fuzzy-only"}
        if strategy not in strategies:
            raise ValueError(f"strategy must be one of {sorted(strategies)}")
        started = time.monotonic()
        top_medicine_name: str | None = None
        top_medicine_id: str | None = None
        retrieval_sources: list[str] = []
        retrieval_score: float | None = None
        candidates_by_id: dict[str, dict[str, Any]] = {}
        named_product_lookup = self._is_named_product_lookup(user_query)
        field_search = self._is_catalog_field_search(user_query)
        advice_request = self._is_advice_request(user_query)
        exact = self._exact_named_medicine(user_query) if strategy in {"hybrid", "exact-fuzzy-only"} else None
        if exact:
            top_medicine_name, top_medicine_id = exact["name"], exact.get("medicine_id")
            retrieval_sources, retrieval_score = ["exact_name"], 1.0

        # Follow-ups remain anchored to the selected medicine unless the new query names another.
        if not top_medicine_name and preferred_medicine_id:
            preferred_rows = self.db.query("""
                MATCH (m:Medicine {medicine_id:$medicine_id})
                RETURN m.name AS name, m.medicine_id AS medicine_id
                LIMIT 1
            """, {"medicine_id": preferred_medicine_id}, db="neo4j") or []
            if preferred_rows:
                top_medicine_name, top_medicine_id = preferred_rows[0]["name"], preferred_rows[0]["medicine_id"]
                retrieval_sources, retrieval_score = ["conversation_context"], 1.0
        if top_medicine_name:
            identity = top_medicine_id or top_medicine_name
            candidates_by_id[identity] = {
                "medicine_id": top_medicine_id, "name": top_medicine_name,
                "rrf_score": retrieval_score or 1.0, "sources": retrieval_sources,
            }

        condition = None if named_product_lookup else self._extract_condition_from_query(user_query)
        condition_rows = self.condition_candidates(condition, limit=200) if condition else []
        product_text = self._product_query_text(user_query) if named_product_lookup else user_query
        fulltext_rows = self.fulltext_search(product_text, limit=20) if strategy in {"fulltext-only", "hybrid", "hybrid-no-exact"} else []
        manufacturer_intent = bool(re.search(r"\b(?:manufacturer|made\s+by|from)\b", user_query, re.I)) and field_search
        manufacturer_rows = self._manufacturer_candidates(user_query, limit=500) if manufacturer_intent else []
        field_rows = self._catalog_field_candidates(user_query, limit=500) if field_search else []
        if condition and manufacturer_intent:
            combined_rows = self._condition_manufacturer_candidates(condition, user_query, limit=50)
            if combined_rows:
                condition_rows, manufacturer_rows = combined_rows, combined_rows
        if condition_rows and manufacturer_rows:
            manufacturer_ids = {row.get("medicine_id") for row in manufacturer_rows}
            intersection = [row for row in condition_rows if row.get("medicine_id") in manufacturer_ids]
            if intersection:
                condition_rows = intersection
                manufacturer_rows = [row for row in manufacturer_rows if row.get("medicine_id") in {item.get("medicine_id") for item in intersection}]
        fuzzy_product = self._fuzzy_product_match(user_query, fulltext_rows or None) if named_product_lookup and not exact else exact

        if not top_medicine_name and strategy == "exact-fuzzy-only":
            if fuzzy_product:
                top_medicine_name, top_medicine_id = fuzzy_product["name"], fuzzy_product.get("medicine_id")
                retrieval_sources, retrieval_score = ["fuzzy_name"], fuzzy_product.get("name_similarity")
                candidates_by_id = {top_medicine_id or top_medicine_name: {
                    **fuzzy_product, "rrf_score": retrieval_score, "sources": retrieval_sources,
                }}
        elif not top_medicine_name:
            rankings = []
            if strategy in {"condition-only", "hybrid", "hybrid-no-exact"} and condition_rows:
                rankings.append(("condition_graph", condition_rows, 2.0 if strategy != "condition-only" else 1.0))
            if strategy in {"fulltext-only", "hybrid", "hybrid-no-exact"} and fulltext_rows:
                rankings.append(("fulltext", fulltext_rows, 1.0))
            if strategy in {"hybrid", "hybrid-no-exact"} and manufacturer_rows:
                rankings.append(("manufacturer", manufacturer_rows, 1.0))
            if strategy in {"hybrid", "hybrid-no-exact"} and field_rows:
                rankings.append(("catalog_field", field_rows, 1.0))
            if strategy in {"vector-only", "hybrid", "hybrid-no-exact"}:
                vector_rows = self.vector_similarity_search(user_query, limit=20)
                rankings.append(("vector", vector_rows, 1.0))
            if rankings:
                ranked = reciprocal_rank_fusion(rankings, rank_constant=0 if strategy == "vector-only" else 60, limit=20)
                candidates_by_id = {item.get("medicine_id") or item.get("name"): item for item in ranked}

        ranked_candidates = sorted(
            candidates_by_id.values(),
            key=lambda item: (-item["rrf_score"], (item["name"] or "").casefold()),
        )
        if strategy in {"hybrid", "hybrid-no-exact"} and (field_search or manufacturer_intent):
            evidence_ids = {row.get("medicine_id") for row in (manufacturer_rows or field_rows)}
            ranked_candidates = [row for row in ranked_candidates if row.get("medicine_id") in evidence_ids]
        abstention_reason = None
        if not top_medicine_name and ranked_candidates:
            winner = ranked_candidates[0]
            # Conservative gates: an answer request needs an identified product, and a product
            # detail lookup cannot fall through to an unrelated semantic neighbour.
            if advice_request:
                abstention_reason = "medical_advice_request_without_named_product"
            elif named_product_lookup and (
                not fuzzy_product
                or winner.get("medicine_id") != fuzzy_product.get("medicine_id")
            ):
                abstention_reason = "no_exact_or_confident_product_match"
            elif not named_product_lookup and not condition and not field_search:
                abstention_reason = "unsupported_query_type"
            elif manufacturer_intent and winner.get("medicine_id") not in {row.get("medicine_id") for row in manufacturer_rows}:
                abstention_reason = "manufacturer_evidence_mismatch"
            elif field_search and field_rows and winner.get("medicine_id") not in {row.get("medicine_id") for row in field_rows}:
                abstention_reason = "catalog_field_evidence_mismatch"
            elif field_search and not (field_rows or manufacturer_rows):
                abstention_reason = "no_structured_field_evidence"
            elif ("vector" in winner.get("sources", [])
                  and not (set(winner.get("sources", [])) & {"fulltext", "condition_graph"})
                  and winner.get("vector_similarity", 0.0) < MIN_VECTOR_SIMILARITY):
                abstention_reason = "vector_similarity_below_threshold"
            if not abstention_reason:
                top_medicine_name = winner["name"]
                top_medicine_id = winner.get("medicine_id")
                retrieval_sources = winner["sources"]
                retrieval_score = winner["rrf_score"]

        if advice_request:
            top_medicine_name = top_medicine_id = None
        if not top_medicine_name:
            abstention_reason = abstention_reason or ("no_candidates" if not ranked_candidates else "no_confident_catalog_match")
            log_event("retrieval_abstained", route=f"rag_{strategy}", reason=abstention_reason, query_hash=hash_query(user_query))
            return {
                "medicine_found": None, "context": None, "ranked_candidates": ranked_candidates,
                "top_vector_similarity": ranked_candidates[0].get("vector_similarity") if ranked_candidates else None,
                "abstained": True, "abstention_reason": abstention_reason,
            }

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
            "ranked_candidates": ranked_candidates,
            "top_vector_similarity": ranked_candidates[0].get("vector_similarity") if ranked_candidates else None,
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
