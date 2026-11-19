import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

class Neo4jConnection:
    """
    A class to manage the connection to a Neo4j database.
    The connection is configured with environment variables or constructor arguments.
    """
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        uri = uri or os.getenv("NEO4J_URI")
        user = user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        password = password or os.getenv("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            raise ValueError("Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in the environment")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        print("Connected to Neo4j")

    def close(self):
        if self._driver is not None:
            self._driver.close()

    def query(self, query, parameters=None, db=None):
        """Runs a Cypher query and returns the results."""
        assert self._driver is not None, "Driver not initialized!"
        with (self._driver.session(database=db) if db is not None else self._driver.session()) as session:
            return list(session.run(query, parameters or {}))


