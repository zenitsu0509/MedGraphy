"""
Script to migrate Neo4j data from local/Aura to AWS Neo4j instance
"""
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import csv

load_dotenv()

# SOURCE Neo4j (your current database)
SOURCE_URI = os.getenv("SOURCE_NEO4J_URI", "bolt://localhost:7687")
SOURCE_USER = os.getenv("SOURCE_NEO4J_USERNAME", "neo4j")
SOURCE_PASSWORD = os.getenv("SOURCE_NEO4J_PASSWORD", "neo4j")

# TARGET Neo4j (AWS Aura or EC2)
TARGET_URI = os.getenv("TARGET_NEO4J_URI")  # neo4j+s://xxxxx.databases.neo4j.io
TARGET_USER = os.getenv("TARGET_NEO4J_USERNAME", "neo4j")
TARGET_PASSWORD = os.getenv("TARGET_NEO4J_PASSWORD")

CSV_PATH = "data/Medicine_Details.csv"


def clear_database(driver):
    """Clear all nodes and relationships"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("✅ Target database cleared")


def ingest_data_from_csv(driver, csv_path):
    """Ingest data from CSV into Neo4j"""
    print(f"📥 Reading CSV from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        medicines = list(reader)
    
    print(f"Found {len(medicines)} medicines in CSV")
    
    with driver.session() as session:
        for idx, med in enumerate(medicines):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(medicines)}...")
            
            # Create Medicine node
            session.run("""
                MERGE (m:Medicine {name: $name})
                SET m.uses = $uses,
                    m.sideEffects = $side_effects,
                    m.image = $image,
                    m.buyLink = $buy_link
            """, 
                name=med['Medicine Name'],
                uses=med['Uses'],
                side_effects=med['Side_effects'],
                image=med.get('Image URL', ''),
                buy_link=med.get('Buy Link', '')
            )
            
            # Create Manufacturer
            if med.get('Manufacturer'):
                session.run("""
                    MERGE (mfr:Manufacturer {name: $mfr_name})
                    WITH mfr
                    MATCH (m:Medicine {name: $med_name})
                    MERGE (m)-[:MANUFACTURED_BY]->(mfr)
                """,
                    mfr_name=med['Manufacturer'],
                    med_name=med['Medicine Name']
                )
            
            # Create Composition (Active Ingredients)
            if med.get('Composition'):
                ingredients = [ing.strip() for ing in med['Composition'].split('+')]
                for ingredient in ingredients:
                    if ingredient:
                        session.run("""
                            MERGE (ing:ActiveIngredient {name: $ing_name})
                            WITH ing
                            MATCH (m:Medicine {name: $med_name})
                            MERGE (m)-[:CONTAINS]->(ing)
                        """,
                            ing_name=ingredient,
                            med_name=med['Medicine Name']
                        )
            
            # Create Side Effects
            if med.get('Side_effects'):
                side_effects = [se.strip() for se in med['Side_effects'].split(',')]
                for se in side_effects[:5]:  # Limit to top 5
                    if se:
                        session.run("""
                            MERGE (side:SideEffect {name: $se_name})
                            WITH side
                            MATCH (m:Medicine {name: $med_name})
                            MERGE (m)-[:HAS_SIDE_EFFECT]->(side)
                        """,
                            se_name=se,
                            med_name=med['Medicine Name']
                        )
            
            # Create Uses/Conditions
            if med.get('Uses'):
                uses = [u.strip() for u in med['Uses'].split(',')]
                for use in uses[:3]:  # Limit to top 3
                    if use:
                        session.run("""
                            MERGE (cond:Condition {name: $cond_name})
                            WITH cond
                            MATCH (m:Medicine {name: $med_name})
                            MERGE (m)-[:TREATS]->(cond)
                        """,
                            cond_name=use,
                            med_name=med['Medicine Name']
                        )
    
    print("✅ Data ingestion complete!")


def main():
    print("=" * 60)
    print("Neo4j Data Migration Script")
    print("=" * 60)
    
    if not TARGET_URI or not TARGET_PASSWORD:
        print("❌ Error: TARGET_NEO4J_URI and TARGET_NEO4J_PASSWORD must be set in .env")
        print("\nExample .env file:")
        print("TARGET_NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io")
        print("TARGET_NEO4J_PASSWORD=your-password")
        return
    
    print(f"\n📡 Connecting to TARGET Neo4j at {TARGET_URI}")
    target_driver = GraphDatabase.driver(
        TARGET_URI,
        auth=(TARGET_USER, TARGET_PASSWORD)
    )
    
    try:
        target_driver.verify_connectivity()
        print("✅ Connected to target database")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Clear target database
    print("\n⚠️  Clearing target database...")
    clear_database(target_driver)
    
    # Ingest from CSV
    print(f"\n📥 Ingesting data from {CSV_PATH}...")
    ingest_data_from_csv(target_driver, CSV_PATH)
    
    # Verify
    with target_driver.session() as session:
        result = session.run("MATCH (n:Medicine) RETURN count(n) as count")
        count = result.single()['count']
        print(f"\n✅ Migration complete! Total medicines in target: {count}")
    
    target_driver.close()
    print("\n" + "=" * 60)
    print("Migration completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
