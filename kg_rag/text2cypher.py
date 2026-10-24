"""Optional fallback: let an LLM write Cypher for questions no template covers.

Only used when *both* an LLM and a Neo4j database are configured. The query
the model writes is treated as untrusted input:

* it must start with MATCH / OPTIONAL MATCH / WITH / RETURN / CALL {;
* any write or admin keyword is rejected outright;
* a LIMIT is appended if missing;
* it runs in a READ-access session (see :class:`kg_rag.backend.Neo4jBackend`),
  so even a query that slipped past the filter cannot change the graph.
"""

from __future__ import annotations

import re

from . import llm

SCHEMA = """Nodes:
  (:Movie {title: string, released: integer, tagline: string})
  (:Person {name: string, born: integer})
Relationships:
  (:Person)-[:ACTED_IN {roles: list<string>}]->(:Movie)
  (:Person)-[:DIRECTED]->(:Movie)
  (:Person)-[:PRODUCED]->(:Movie)
  (:Person)-[:WROTE]->(:Movie)
  (:Person)-[:REVIEWED {summary: string, rating: integer}]->(:Movie)
  (:Person)-[:FOLLOWS]->(:Person)"""

PROMPT = f"""Translate the user's question into ONE read-only Cypher query for Neo4j 5.
Graph schema:
{SCHEMA}
Return only the query - no explanation, no markdown fences. Use readable column
aliases with AS. Never use CREATE, MERGE, SET, DELETE, REMOVE, DROP or LOAD CSV."""

FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD\s+CSV|FOREACH|GRANT|DENY|REVOKE)\b"
    r"|\bdbms\.|\bapoc\.|\bgds\.",
    re.IGNORECASE,
)
START = re.compile(r"^\s*(OPTIONAL\s+MATCH|MATCH|WITH|RETURN|UNWIND|CALL\s*\{)", re.IGNORECASE)


class UnsafeCypher(ValueError):
    pass


def sanitise(query: str, limit: int = 25) -> str:
    query = re.sub(r"^```(?:cypher)?|```$", "", query.strip(), flags=re.MULTILINE).strip()
    query = query.rstrip(";").strip()
    if ";" in query:
        raise UnsafeCypher("multiple statements")
    if not START.match(query):
        raise UnsafeCypher("query must start with MATCH, WITH, RETURN, UNWIND or CALL {")
    bad = FORBIDDEN.search(query)
    if bad:
        raise UnsafeCypher(f"forbidden keyword: {bad.group(0)}")
    if not re.search(r"\bLIMIT\s+\d+\s*$", query, re.IGNORECASE):
        query += f"\nLIMIT {limit}"
    return query


def generate(question: str) -> str:
    return sanitise(llm.chat(PROMPT, question, max_tokens=300))
