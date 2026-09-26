from neo4j import GraphDatabase
from functools import lru_cache

from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


@lru_cache(maxsize=1)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
