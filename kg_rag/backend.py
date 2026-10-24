"""Graph backends: a real Neo4j database, or the bundled in-memory graph.

Both expose the same two calls:

* ``run_template(template, params)`` -> rows
* ``run_cypher(query, params)``      -> rows   (Neo4j only)

:func:`connect` tries Neo4j when ``NEO4J_URI`` is set and falls back to the
offline graph if the database cannot be reached, recording why. The interface
always says which backend answered, so an offline answer is never presented as
if it came from Neo4j.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .dataset import MovieGraph
from .templates import Template, normalise


class BackendError(RuntimeError):
    pass


@dataclass
class LocalBackend:
    graph: MovieGraph = field(default_factory=MovieGraph)
    name: str = "offline"
    detail: str = "bundled Movies graph held in memory (no Neo4j configured)"

    @property
    def is_neo4j(self) -> bool:
        return False

    def run_template(self, template: Template, params: dict) -> list[dict]:
        return normalise(template, template.local(self.graph, params))

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        raise BackendError("Free-form Cypher needs a Neo4j database - set NEO4J_URI.")

    def stats(self) -> dict:
        return self.graph.stats()


class Neo4jBackend:
    """Thin wrapper over the official driver. Every query runs in a READ session."""

    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase   # imported lazily: offline mode never needs it

        self.uri, self.database = uri, database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.detail = f"{uri} (database '{database}')"

    @property
    def is_neo4j(self) -> bool:
        return True

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        from neo4j import READ_ACCESS

        with self.driver.session(database=self.database, default_access_mode=READ_ACCESS) as s:
            return [record.data() for record in s.run(query, params or {})]

    def run_template(self, template: Template, params: dict) -> list[dict]:
        return normalise(template, self.run_cypher(template.cypher, params))

    def stats(self) -> dict:
        rows = self.run_cypher(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt")
        counts = {r["label"]: r["cnt"] for r in rows}
        rels = self.run_cypher("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt ORDER BY rel_type")
        by_type = {r["rel_type"]: r["cnt"] for r in rels}
        return {"movies": counts.get("Movie", 0), "people": counts.get("Person", 0),
                "relationships": sum(by_type.values()), "by_type": by_type}

    def close(self) -> None:
        self.driver.close()


def connect(prefer_neo4j: bool = True):
    """Return ``(backend, warning)``. Never raises: offline mode is always available."""
    if prefer_neo4j and config.NEO4J_URI:
        try:
            backend = Neo4jBackend(config.NEO4J_URI, config.NEO4J_USER,
                                   config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
            if backend.stats()["movies"] == 0:
                return LocalBackend(), (
                    "Neo4j is reachable but empty - run `python scripts/load_neo4j.py`. "
                    "Using the offline graph meanwhile.")
            return backend, None
        except Exception as exc:  # driver import, auth, network...
            return LocalBackend(), f"Could not reach Neo4j at {config.NEO4J_URI}: {exc}. Using the offline graph."
    return LocalBackend(), None
