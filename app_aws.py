import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from neo4j import GraphDatabase
from dotenv import load_dotenv 
import os
import boto3
import tempfile

load_dotenv()

st.set_page_config(page_title="Medicine GraphRAG AI", layout="wide")

# AWS Configuration
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "medgraphy-faiss-db")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Other configs
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI") or st.secrets.get("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") or st.secrets.get("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or st.secrets.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

FAISS_INDEX_S3_KEY = "db/medicine_embeddings.index"
METADATA_S3_KEY = "db/metadata.json"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------
#           LOAD MODELS & DATABASES FROM S3
# ---------------------------------------------------------

@st.cache_resource
def load_faiss_from_s3():
    """Download FAISS index from S3 and load it"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # Create temp file for FAISS index
    with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        st.info(f"📥 Downloading FAISS index from S3: s3://{AWS_S3_BUCKET}/{FAISS_INDEX_S3_KEY}")
        s3.download_file(AWS_S3_BUCKET, FAISS_INDEX_S3_KEY, tmp_path)
        index = faiss.read_index(tmp_path)
        st.success("✅ FAISS index loaded from S3")
        return index
    except Exception as e:
        st.error(f"❌ Failed to load FAISS from S3: {str(e)}")
        raise
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@st.cache_resource
def load_metadata_from_s3():
    """Download metadata JSON from S3"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    try:
        st.info(f"📥 Downloading metadata from S3: s3://{AWS_S3_BUCKET}/{METADATA_S3_KEY}")
        obj = s3.get_object(Bucket=AWS_S3_BUCKET, Key=METADATA_S3_KEY)
        metadata = json.loads(obj['Body'].read().decode('utf-8'))
        st.success("✅ Metadata loaded from S3")
        return metadata
    except Exception as e:
        st.error(f"❌ Failed to load metadata from S3: {str(e)}")
        raise

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_llm():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_neo4j():
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=120
    )
    # Test the connection
    driver.verify_connectivity()
    return driver


# Load all resources
try:
    faiss_index = load_faiss_from_s3()
    metadata = load_metadata_from_s3()
    st.sidebar.success("✅ FAISS loaded from S3")
except Exception as e:
    st.sidebar.error(f"❌ FAISS/S3 Error: {str(e)}")
    st.error("Cannot load FAISS database from S3. Check AWS credentials and bucket configuration.")
    st.stop()

embedder = load_embedder()
groq_client = load_llm()

# Load Neo4j with error handling
try:
    neo4j_driver = load_neo4j()
    st.sidebar.success("✅ Connected to Neo4j")
except Exception as e:
    st.sidebar.error(f"❌ Neo4j Connection Failed: {str(e)}")
    st.error(f"Database connection error. Please check your Neo4j credentials: {str(e)}")
    neo4j_driver = None


# ---------------------------------------------------------
#       GRAPH EXPANSION — FETCH RELATED NODES
# ---------------------------------------------------------

def get_graph_info(drug_name):
    if neo4j_driver is None:
        return {}
    
    # Use case-insensitive matching
    query = """
    MATCH (m:Medicine)
    WHERE toLower(m.name) = toLower($name)
    OPTIONAL MATCH (m)-[r]->(n)
    WITH type(r) AS rel_type, n.name AS target_name
    WHERE rel_type IS NOT NULL
    RETURN rel_type AS relation, target_name AS value
    LIMIT 200
    """
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, name=drug_name).data()
    except Exception as e:
        print(f"Graph query error: {e}")
        return {}

    graph_dict = {}
    for row in result:
        relation = row.get("relation")
        value = row.get("value")
        if relation and value:
            graph_dict.setdefault(relation, []).append(value)

    return graph_dict


# ---------------------------------------------------------
#       DIRECT NEO4J SEARCH (Graph-based)
# ---------------------------------------------------------

def search_neo4j_directly(query, limit=10):
    """
    Search Neo4j directly for medicines, conditions, side effects, or ingredients
    based on the query keywords.
    """
    if neo4j_driver is None:
        return {"medicines": [], "conditions": [], "side_effects": [], "ingredients": []}
    
    results = {
        "medicines": [],
        "conditions": [],
        "side_effects": [],
        "ingredients": []
    }
    
    # Extract keywords from query (simple approach)
    query_lower = query.lower()
    
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            # Search medicines by name or composition containing query terms
            med_query = """
            MATCH (m:Medicine)
            WHERE toLower(m.name) CONTAINS $query 
               OR toLower(m.composition) CONTAINS $query
               OR toLower(m.uses_text) CONTAINS $query
            RETURN m.name AS name, m.composition AS composition, 
                   m.uses_text AS uses, m.side_effects_text AS side_effects,
                   m.excellent_review_pct AS excellent_review
            ORDER BY m.excellent_review_pct DESC
            LIMIT $limit
            """
            med_results = session.run(med_query, query=query_lower, limit=limit).data()
            results["medicines"] = med_results
            
            # Search conditions that match query
            cond_query = """
            MATCH (c:Condition)<-[:TREATS]-(m:Medicine)
            WHERE toLower(c.name) CONTAINS $query
            RETURN c.name AS condition, collect(DISTINCT m.name)[0..5] AS treating_medicines
            LIMIT 5
            """
            cond_results = session.run(cond_query, query=query_lower).data()
            results["conditions"] = cond_results
            
            # Search side effects that match query
            se_query = """
            MATCH (s:SideEffect)<-[:HAS_SIDE_EFFECT]-(m:Medicine)
            WHERE toLower(s.name) CONTAINS $query
            RETURN s.name AS side_effect, collect(DISTINCT m.name)[0..5] AS medicines_with_effect
            LIMIT 5
            """
            se_results = session.run(se_query, query=query_lower).data()
            results["side_effects"] = se_results
            
            # Search ingredients that match query
            ing_query = """
            MATCH (i:ActiveIngredient)<-[:CONTAINS_INGREDIENT]-(m:Medicine)
            WHERE toLower(i.name) CONTAINS $query
            RETURN i.name AS ingredient, collect(DISTINCT m.name)[0..10] AS medicines_containing
            LIMIT 5
            """
            ing_results = session.run(ing_query, query=query_lower).data()
            results["ingredients"] = ing_results
            
    except Exception as e:
        print(f"Neo4j direct search error: {e}")
    
    return results


# ---------------------------------------------------------
#            SEMANTIC SEARCH (FAISS)
# ---------------------------------------------------------

def semantic_search(query, top_k=5):
    query_emb = embedder.encode(query).astype("float32")

    distances, indices = faiss_index.search(
        np.array([query_emb]), top_k
    )

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    return results


# ---------------------------------------------------------
#            LLM ANSWER USING GROQ
# ---------------------------------------------------------

def answer_with_groq(query, faiss_results, graph_expansion, neo4j_direct_results):
    system_prompt = """
    You are a medical question answering assistant with access to TWO data sources:
    
    1. **FAISS Vector Database**: Semantic similarity search results - good for finding medicines 
       related to the query meaning, even if exact keywords don't match.
    
    2. **Neo4j Graph Database**: 
       - Direct search results: Exact matches for medicines, conditions, side effects, ingredients
       - Graph expansion: Relationships like TREATS, HAS_SIDE_EFFECT, CONTAINS_INGREDIENT, MANUFACTURED_BY
    
    Your task:
    - Analyze BOTH data sources
    - Decide which source is more relevant for the specific question
    - You can use BOTH sources if they provide complementary information
    - For specific medicine queries → prioritize Neo4j direct matches
    - For general symptom/condition queries → combine FAISS semantics + Neo4j graph relationships
    - For side effect queries → prioritize Neo4j graph data (HAS_SIDE_EFFECT relationships)
    - For ingredient queries → prioritize Neo4j graph data (CONTAINS_INGREDIENT relationships)
    
    Rules:
    - Never hallucinate facts - use ONLY the provided context
    - If data is conflicting, prefer Neo4j graph data (more structured)
    - Clearly cite which source provided the information when helpful
    - Be concise but comprehensive
    """

    # Build context from FAISS metadata
    faiss_text = "=== FAISS VECTOR SEARCH RESULTS ===\n"
    if faiss_results:
        for item in faiss_results:
            faiss_text += f"""
Medicine: {item.get('name', 'N/A')}
Uses: {item.get('uses', 'N/A')}
Side Effects: {item.get('side_effects', 'N/A')}
Manufacturer: {item.get('manufacturer', 'N/A')}
---
"""
    else:
        faiss_text += "No FAISS results found.\n"

    # Build graph expansion info
    graph_text = "\n=== NEO4J GRAPH EXPANSION (Relationships) ===\n"
    has_graph_data = False
    for medicine, relations in graph_expansion.items():
        if relations:
            has_graph_data = True
            graph_text += f"\n📊 Graph Data for '{medicine}':\n"
            for rel, vals in relations.items():
                rel_readable = rel.replace("_", " ").title()
                graph_text += f"  • {rel_readable}: {', '.join(vals[:10])}\n"
    if not has_graph_data:
        graph_text += "No graph expansion data found.\n"

    # Build Neo4j direct search results
    neo4j_text = "\n=== NEO4J DIRECT SEARCH RESULTS ===\n"
    has_neo4j_data = False
    
    if neo4j_direct_results.get("medicines"):
        has_neo4j_data = True
        neo4j_text += "\n🔍 Matching Medicines:\n"
        for med in neo4j_direct_results["medicines"][:5]:
            neo4j_text += f"  • {med.get('name', 'N/A')}\n"
            neo4j_text += f"    Uses: {med.get('uses', 'N/A')[:200]}...\n" if med.get('uses') else ""
            neo4j_text += f"    Side Effects: {med.get('side_effects', 'N/A')[:200]}...\n" if med.get('side_effects') else ""
    
    if neo4j_direct_results.get("conditions"):
        has_neo4j_data = True
        neo4j_text += "\n🏥 Matching Conditions:\n"
        for cond in neo4j_direct_results["conditions"]:
            neo4j_text += f"  • {cond.get('condition', 'N/A')}\n"
            neo4j_text += f"    Treating Medicines: {', '.join(cond.get('treating_medicines', []))}\n"
    
    if neo4j_direct_results.get("side_effects"):
        has_neo4j_data = True
        neo4j_text += "\n⚠️ Matching Side Effects:\n"
        for se in neo4j_direct_results["side_effects"]:
            neo4j_text += f"  • {se.get('side_effect', 'N/A')}\n"
            neo4j_text += f"    Found in: {', '.join(se.get('medicines_with_effect', []))}\n"
    
    if neo4j_direct_results.get("ingredients"):
        has_neo4j_data = True
        neo4j_text += "\n💊 Matching Ingredients:\n"
        for ing in neo4j_direct_results["ingredients"]:
            neo4j_text += f"  • {ing.get('ingredient', 'N/A')}\n"
            neo4j_text += f"    Found in: {', '.join(ing.get('medicines_containing', [])[:5])}\n"
    
    if not has_neo4j_data:
        neo4j_text += "No direct Neo4j matches found.\n"

    full_prompt = f"""
{system_prompt}

📝 USER QUERY: {query}

{faiss_text}
{graph_text}
{neo4j_text}

Based on the above data sources, provide a comprehensive answer. Indicate which data source(s) you primarily used.
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
#                     STREAMLIT UI
# ---------------------------------------------------------

st.title("💊 Medicine GraphRAG AI (AWS Edition)")
st.markdown("**Dual Database Search: FAISS Vector DB (S3) + Neo4j Graph DB (Aura) + LLM Reasoning (Groq)**")

# Show AWS info and status in sidebar
with st.sidebar:
    st.markdown("### ☁️ AWS Configuration")
    st.code(f"""
S3 Bucket: {AWS_S3_BUCKET}
Region: {AWS_REGION}
Neo4j: {NEO4J_URI}
    """)
    
    # Database status
    st.markdown("### 📊 Database Status")
    if neo4j_driver:
        st.success("✅ Connected to Neo4j Aura")
    else:
        st.error("❌ Neo4j Connection Failed")
    st.info("✅ FAISS loaded from S3")

# Initialize session state for query
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# Main query input
query = st.text_input(
    "Enter your medical query:",
    placeholder="e.g., What are the side effects of paracetamol?",
    value=st.session_state.current_query
)

# Action buttons
col1, col2 = st.columns([3, 1])
with col1:
    search_btn = st.button("🔍 Search Both Databases", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

# Handle clear button
if clear_btn:
    st.rerun()

# Process search
if search_btn and query.strip():
    with st.spinner("🔍 Searching databases..."):
        # Step 1: FAISS Semantic Search
        faiss_results = semantic_search(query)
        
        # Step 2: Neo4j Direct Search
        neo4j_direct_results = search_neo4j_directly(query)
        
        # Step 3: Graph expansion for FAISS results
        graph_expansion = {}
        for r in faiss_results:
            graph_expansion[r["name"]] = get_graph_info(r["name"])
        
        # Step 4: Generate LLM answer using all sources
        with st.spinner("🤖 Generating AI answer..."):
            answer = answer_with_groq(query, faiss_results, graph_expansion, neo4j_direct_results)
    
    # Display AI Answer FIRST (most important)
    st.markdown("---")
    st.markdown("### 🩺 AI Answer (Using Both Databases)")
    st.success(answer)
    
    st.markdown("---")
    
    # Display database results in expandable sections
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔍 FAISS Vector Search Results", expanded=False):
            if faiss_results:
                for r in faiss_results:
                    st.markdown(f"**{r['name']}**")
                    st.markdown(f"- **Uses:** {r.get('uses', 'N/A')[:150]}...")
                    st.markdown(f"- **Side Effects:** {r.get('side_effects', 'N/A')[:100]}...")
                    st.markdown("---")
            else:
                st.info("No FAISS results found")
    
    with col2:
        with st.expander("🧬 Neo4j Graph Database Results", expanded=False):
            # Direct matches
            if neo4j_direct_results.get("medicines"):
                st.markdown("**📋 Direct Medicine Matches:**")
                for med in neo4j_direct_results["medicines"][:5]:
                    st.markdown(f"- {med.get('name', 'N/A')}")
                st.markdown("")
            
            if neo4j_direct_results.get("conditions"):
                st.markdown("**🏥 Matching Conditions:**")
                for cond in neo4j_direct_results["conditions"]:
                    medicines = ', '.join(cond.get('treating_medicines', [])[:3])
                    st.markdown(f"- {cond.get('condition', 'N/A')}: {medicines}")
                st.markdown("")
            
            if neo4j_direct_results.get("ingredients"):
                st.markdown("**💊 Matching Ingredients:**")
                for ing in neo4j_direct_results["ingredients"]:
                    medicines = ', '.join(ing.get('medicines_containing', [])[:3])
                    st.markdown(f"- {ing.get('ingredient', 'N/A')}: {medicines}")
                st.markdown("")
            
            if neo4j_direct_results.get("side_effects"):
                st.markdown("**⚠️ Matching Side Effects:**")
                for se in neo4j_direct_results["side_effects"]:
                    medicines = ', '.join(se.get('medicines_with_effect', [])[:3])
                    st.markdown(f"- {se.get('side_effect', 'N/A')}: {medicines}")
                st.markdown("")
            
            # Graph expansion
            if graph_expansion:
                st.markdown("**🔗 Graph Relationships:**")
                st.json(graph_expansion)
            
            if not any([
                neo4j_direct_results.get("medicines"),
                neo4j_direct_results.get("conditions"),
                neo4j_direct_results.get("ingredients"),
                neo4j_direct_results.get("side_effects"),
                graph_expansion
            ]):
                st.info("No Neo4j results found")

elif search_btn and not query.strip():
    st.warning("⚠️ Please enter a query.")

# Examples section
st.markdown("---")
st.markdown("### 📝 Example Queries")
example_queries = [
    "What is the best medicine for acidity?",
    "Show me medicines for headache",
    "What are the side effects of paracetamol?",
    "Suggest medicine for cold and fever",
    "Find medicines containing ibuprofen",
    "What treats hypertension?"
]

cols = st.columns(3)
for idx, example in enumerate(example_queries):
    with cols[idx % 3]:
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            st.session_state.current_query = example
            st.rerun()
