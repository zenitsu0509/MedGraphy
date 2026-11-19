"""FastAPI application for MEDGraphy's browser UI and JSON API."""

from __future__ import annotations

import os
import json
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(".env.local")
load_dotenv()

from graph_db import Neo4jConnection  # noqa: E402
from graph_rag_query import GraphQueryEngine, VECTOR_INDEX_NAME  # noqa: E402
from llm_chains import generate_rag_response, get_groq_client  # noqa: E402
from monitoring import hash_query, log_event, source_freshness  # noqa: E402
from data_pipeline import normalize_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"
AYURVEDA_IMAGES = ROOT / "data" / "ayurveda" / "Medicinal Leaf Dataset" / "Segmented Medicinal Leaf Images"


class RagRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db = Neo4jConnection()
        app.state.neo4j_connection = db
        engine = GraphQueryEngine(db=db)
        app.state.query_engine = engine
        try:
            app.state.groq_client = get_groq_client()
        except Exception as exc:
            app.state.groq_client = None
            log_event("api_llm_unavailable", error_type=type(exc).__name__)
        log_event("api_started", model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
        yield
    except Exception as exc:
        log_event("api_startup_failed", error_type=type(exc).__name__)
        raise
    finally:
        engine = getattr(app.state, "query_engine", None)
        if engine is not None:
            engine.db.close()
        elif getattr(app.state, "neo4j_connection", None) is not None:
            app.state.neo4j_connection.close()
        log_event("api_stopped")


app = FastAPI(
    title="MEDGraphy API",
    version="1.0.0",
    description="Medicine catalog and evidence-aware graph retrieval API.",
    lifespan=lifespan,
)
if AYURVEDA_IMAGES.is_dir():
    app.mount("/botanical-images", StaticFiles(directory=AYURVEDA_IMAGES), name="botanical-images")


@lru_cache(maxsize=1)
def _plant_image_classes() -> dict[str, tuple[str, str]]:
    """Index folder display names and their parenthetical common names."""
    result = {}
    if not AYURVEDA_IMAGES.is_dir():
        return result
    for folder in AYURVEDA_IMAGES.iterdir():
        if not folder.is_dir():
            continue
        images = sorted(path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not images:
            continue
        labels = [folder.name]
        match = re.search(r"\(([^()]*)\)\s*$", folder.name)
        if match:
            labels.append(match.group(1))
        for label in labels:
            result[normalize_key(label)] = (folder.name, images[0])
    return result


def _engine(request: Request) -> GraphQueryEngine:
    engine = getattr(request.app.state, "query_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Backend is still starting")
    return engine


def _medicine_card(engine: GraphQueryEngine, medicine_id: str) -> dict[str, Any] | None:
    rows = engine.db.query("""
        MATCH (m:Medicine {medicine_id:$medicine_id})
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
        RETURN m.name AS name, m.medicine_id AS medicine_id,
               m.composition AS composition, m.uses_text AS uses_text,
               m.side_effects_text AS side_effects_text, m.image_url AS image_url,
               m.excellent_review_pct AS excellent_review_pct,
               m.average_review_pct AS average_review_pct,
               m.poor_review_pct AS poor_review_pct,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT s.name) AS side_effects,
               collect(DISTINCT i.name) AS ingredients,
               head(collect(DISTINCT mf.name)) AS manufacturer,
               properties(m) AS medicine_props
    """, {"medicine_id": medicine_id}, db="neo4j")
    if not rows:
        return None
    card = dict(rows[0])
    props = card.pop("medicine_props", {})
    card.update({
        "source_id": props.get("source_id"),
        "source_name": props.get("source_name"),
        "source_url": props.get("source_url"),
        "source_date": props.get("source_date"),
        "retrieved_at": props.get("source_retrieved_at"),
        "license": props.get("source_license"),
        "jurisdiction": props.get("jurisdiction"),
        "clinical_authority": props.get("clinical_authority"),
        "provenance_status": props.get("provenance_status"),
        "source_conflict_fields": props.get("source_conflict_fields", []),
    })
    return card


@app.get("/api/health")
def health(request: Request):
    engine = _engine(request)
    try:
        engine.db.query("RETURN 1 AS ok", db="neo4j")
        index_rows = engine.db.query("""
            SHOW VECTOR INDEXES YIELD name, state, populationPercent
            WHERE name=$index_name
            RETURN state, populationPercent
        """, {"index_name": VECTOR_INDEX_NAME}, db="neo4j")
        index = dict(index_rows[0]) if index_rows else {"state": "missing", "populationPercent": 0}
        return {"status": "ok", "database": "connected", "vector_index": index}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Neo4j is unavailable") from exc


@app.get("/api/ayurveda/search")
def search_ayurveda(
    request: Request,
    q: str = Query(min_length=2, max_length=120),
    field: str = Query(default="all", pattern="^(all|herb|problem|symptom)$"),
    limit: int = Query(default=20, ge=1, le=30),
):
    """Search isolated, provenance-carrying Ayurveda preview records in Neo4j."""
    query_key = normalize_key(q)
    engine = _engine(request)
    try:
        rows = engine.db.query("""
            MATCH (e:AyurvedaEntry)
            WHERE ($field IN ['all','problem'] AND e.normalized_problem CONTAINS $q)
               OR ($field IN ['all','symptom'] AND e.normalized_symptoms CONTAINS $q)
               OR ($field IN ['all','herb'] AND e.normalized_medicines CONTAINS $q)
            OPTIONAL MATCH (e)-[:MENTIONS_HERB]->(h:AyurvedicHerb)
            RETURN properties(e) AS entry, collect(DISTINCT h.name) AS herbs
            ORDER BY entry.source_row_number
            LIMIT $limit
        """, {"field": field, "q": query_key, "limit": limit}, db="neo4j")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Ayurveda graph search failed; run the Ayurveda ingestion command first") from exc
    matched = []
    image_classes = _plant_image_classes()
    for row in rows:
        entry = dict(row["entry"])
        herbs = list(row["herbs"] or [])
        images = []
        for herb in herbs:
            match = image_classes.get(normalize_key(herb))
            if match:
                folder, filename = match
                images.append({"herb": herb, "class_name": folder,
                    "url": "/botanical-images/" + quote(folder, safe="") + "/" + quote(filename, safe="")})
        original = {}
        try:
            original = json.loads(entry.get("source_record_json", "{}"))
        except (TypeError, ValueError):
            pass
        matched.append({
            "record_id": entry.get("record_id"), "source_row_id": entry.get("source_row_id"),
            "problem": entry.get("problem"), "symptoms": entry.get("symptoms"), "herbs": herbs,
            "remedies": original.get("Remedies"), "contraindications": original.get("Contraindications"),
            "preventive_advice": original.get("Preventive Advice"),
            "dosha_type": original.get("Dosha Type"), "body_system": original.get("Body System"),
            "chronic_acute": original.get("Chronic/Acute"), "treatment_type": original.get("Treatment Type"),
            "seasonal_suitability": original.get("Seasonal Suitability"),
            "classical_texts": original.get("Classical Texts"),
            "modern_evidence_text": original.get("Modern Evidence"),
            "institutional_endorsements": original.get("Institutional Endorsements"),
            "authentication_notes": original.get("Authentication Notes"),
            "confidence_value": entry.get("confidence_value"), "source_reference": entry.get("source_reference"),
            "authentication_score": entry.get("authentication_score"),
            "source_id": entry.get("source_id"), "source_sha256": entry.get("source_sha256"),
            "jurisdiction": entry.get("jurisdiction", "unknown"), "source_date": entry.get("source_date", "unknown"),
            "retrieved_at": entry.get("retrieved_at", "unknown"), "ingested_at": entry.get("ingested_at"), "license": entry.get("license", "unknown"),
            "provenance_status": entry.get("provenance_status", "unverified_preview"), "images": images,
        })
    log_event("ayurveda_preview_search", query_hash=hash_query(q), field=field, hit_count=len(matched))
    return {
        "query": q,
        "field": field,
        "results": matched,
        "dataset_status": "unverified_preview",
        "notice": "Imported dataset text is unverified and must not be treated as medical advice.",
    }


@app.get("/api/medicines/search")
def search_medicines(request: Request, q: str = Query(min_length=2, max_length=120), limit: int = Query(default=10, ge=1, le=30)):
    engine = _engine(request)
    try:
        rows = engine.db.query("""
            MATCH (m:Medicine)
            WHERE toLower(m.name) CONTAINS toLower($q)
            OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
            RETURN m.medicine_id AS medicine_id, m.name AS name,
                   m.composition AS composition, head(collect(DISTINCT mf.name)) AS manufacturer
            ORDER BY CASE WHEN toLower(m.name)=toLower($q) THEN 0 ELSE 1 END,
                     m.name, m.composition
            LIMIT $limit
        """, {"q": q.strip(), "limit": limit}, db="neo4j")
        return {"results": [dict(row) for row in rows]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Medicine search failed") from exc


@app.get("/api/medicines/{medicine_id}")
def medicine_detail(medicine_id: str, request: Request):
    engine = _engine(request)
    try:
        card = _medicine_card(engine, medicine_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Medicine lookup failed") from exc
    if card is None:
        raise HTTPException(status_code=404, detail="Medicine not found")
    return card


@app.get("/api/conditions/search")
def search_condition(request: Request, q: str = Query(min_length=2, max_length=120), limit: int = Query(default=20, ge=1, le=30)):
    engine = _engine(request)
    query = q.strip()
    corrected_query = None
    try:
        names = engine.reverse_lookup(query)[:limit]
        if not names:
            corrected_query = engine.suggest_condition_query(query)
            if corrected_query and corrected_query.casefold() != query.casefold():
                names = engine.reverse_lookup(corrected_query)[:limit]
        rows = engine.db.query("""
            MATCH (m:Medicine) WHERE m.name IN $names
            OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
            OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            RETURN m.medicine_id AS medicine_id, m.name AS name,
                   m.composition AS composition, m.image_url AS image_url,
                   m.uses_text AS uses_text, m.side_effects_text AS side_effects_text,
                   head(collect(DISTINCT mf.name)) AS manufacturer,
                   collect(DISTINCT c.name) AS conditions,
                   collect(DISTINCT s.name) AS side_effects,
                   collect(DISTINCT i.name) AS ingredients,
                   properties(m) AS medicine_props
            ORDER BY m.name, m.composition
            LIMIT $limit
        """, {"names": names, "limit": limit}, db="neo4j") if names else []
        result = []
        for row in rows:
            item = dict(row)
            props = item.pop("medicine_props", {})
            item.update({
                "source_id": props.get("source_id"), "source_name": props.get("source_name"),
                "source_url": props.get("source_url"), "source_date": props.get("source_date"),
                "retrieved_at": props.get("source_retrieved_at"), "license": props.get("source_license"),
                "jurisdiction": props.get("jurisdiction"), "provenance_status": props.get("provenance_status"),
                "source_conflict_fields": props.get("source_conflict_fields", []),
            })
            result.append(item)
        return {"query": query, "corrected_query": corrected_query, "results": result}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Condition search failed") from exc


@app.get("/api/medicines/{medicine_id}/shared-ingredients")
def shared_ingredients(medicine_id: str, request: Request):
    engine = _engine(request)
    try:
        rows = engine.db.query("""
            MATCH (m:Medicine {medicine_id:$medicine_id})-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            MATCH (o:Medicine)-[:CONTAINS_INGREDIENT]->(i)
            WHERE o <> m
            RETURN o.medicine_id AS medicine_id, o.name AS medicine,
                   i.name AS shared_ingredient
            ORDER BY o.name LIMIT 30
        """, {"medicine_id": medicine_id}, db="neo4j")
        return {"meaning": "Shared ingredient only; this does not establish an interaction.", "results": [dict(row) for row in rows]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Ingredient lookup failed") from exc


@app.get("/api/medicines/{medicine_id}/interactions")
def known_interactions(medicine_id: str, request: Request):
    engine = _engine(request)
    try:
        rows = engine.db.query("""
            MATCH (m:Medicine {medicine_id:$medicine_id})-[r]-(o:Medicine)
            WHERE type(r)='HAS_KNOWN_INTERACTION'
            RETURN o.medicine_id AS medicine_id, o.name AS medicine,
                   properties(r) AS evidence
            LIMIT 30
        """, {"medicine_id": medicine_id}, db="neo4j")
        return {"results": [{**dict(row), **(row.get("evidence") or {})} for row in rows]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Interaction lookup failed") from exc


@app.post("/api/rag")
def rag(payload: RagRequest, request: Request):
    engine = _engine(request)
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank")
    try:
        retrieved = engine.retrieve_context_for_rag(query)
    except Exception as exc:
        log_event("api_rag_retrieval_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail="Retrieval failed") from exc
    if not retrieved or not retrieved.get("context"):
        return {"answer": "No relevant catalog record was found.", "medicine": None, "sources": []}
    if request.app.state.groq_client is None:
        raise HTTPException(status_code=503, detail="Answer generation is unavailable; configure GROQ_API_KEY")
    context = retrieved["context"]
    try:
        answer = generate_rag_response(query, context, request.app.state.groq_client)
    except Exception as exc:
        log_event("api_rag_generation_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Answer generation failed") from exc
    return {
        "answer": answer,
        "medicine": {"medicine_id": context.get("medicine_id"), "name": retrieved.get("medicine_found")},
        "sources": [{
            "ref": "S1", "source_id": context.get("source_id"),
            "name": context.get("source_name"), "url": context.get("source_url"),
            "source_date": context.get("source_date"), "retrieved_at": context.get("retrieved_at"),
            "license": context.get("license"), "jurisdiction": context.get("jurisdiction"),
            "freshness": source_freshness(context.get("retrieved_at")),
        }],
    }


@app.get("/", include_in_schema=False)
def homepage():
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="Frontend file is missing")
    return FileResponse(FRONTEND, media_type="text/html", headers={"Cache-Control": "no-store"})
