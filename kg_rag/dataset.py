"""The Neo4j Movies dataset: parsing, storage and an in-memory graph.

Neo4j ships the Movies graph as a Cypher script (``:play movies``). That script
is kept verbatim in ``data/movies/movies.cypher`` and parsed here into a plain
JSON file of nodes and relationships. The JSON is what both backends load:

* ``scripts/load_neo4j.py`` pushes it into Neo4j with ``UNWIND ... MERGE``;
* :class:`MovieGraph` holds it in memory for the offline backend.

Using one file for both guarantees the two backends answer over identical data.

The parser keeps a per-statement variable scope, exactly as Cypher does: a
variable bound in one statement is not visible after the next semicolon. An
edge whose endpoints are not bound in its own statement is counted in
``dropped_unbound_edges`` rather than guessed at; for the shipped script that
count is zero, and the result matches Neo4j's published totals (38 movies,
133 people, 253 relationships).
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import MOVIES_CYPHER, MOVIES_JSON

NODE_RE = re.compile(r"(?:MERGE|MATCH)\s*\((\w+):(\w+)\s*\{(.+?)\}\)\s*(?:ON CREATE|$)")
SET_RE = re.compile(r"ON CREATE SET\s+(.*)$")
REL_RE = re.compile(r"MERGE\s*\((\w+)\)-\[:(\w+)(?:\s*\{(.*)\})?\]->\((\w+)\)")


def _literal(text: str):
    """Turn a Cypher literal ('x', "x", 12, ['a','b']) into a Python value."""
    return ast.literal_eval(text.strip())


def _props(text: str) -> dict:
    """Parse ``key:value, key:value`` - values may contain commas inside quotes."""
    props: dict = {}
    for match in re.finditer(r"(\w+)\s*:\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\[[^\]]*\]|-?\d+)", text):
        props[match.group(1)] = _literal(match.group(2))
    return props


def parse_cypher(path: Path = MOVIES_CYPHER) -> dict:
    """Parse the Movies Cypher script into ``{"nodes": [...], "rels": [...]}``."""
    nodes: dict[tuple[str, str], dict] = {}
    rels: list[dict] = []
    seen_rels: set[tuple] = set()
    dropped = 0

    for statement in path.read_text(encoding="utf-8").split(";"):
        scope: dict[str, tuple[str, str]] = {}          # variable -> node key
        for line in statement.splitlines():
            line = line.strip()
            if not line or line.startswith("CREATE CONSTRAINT"):
                continue
            node = NODE_RE.match(line)
            if node:
                var, label, raw = node.groups()
                inline = _props(raw)
                key_prop = "title" if label == "Movie" else "name"
                key = (label, str(inline[key_prop]))
                record = nodes.setdefault(key, {"label": label})
                record.update(inline)
                set_clause = SET_RE.search(line)
                if set_clause:
                    for assignment in re.split(r",\s*(?=\w+\.\w+\s*=)", set_clause.group(1)):
                        prop_path, value = assignment.split("=", 1)
                        record[prop_path.split(".", 1)[1].strip()] = _literal(value)
                scope[var] = key
                continue
            rel = REL_RE.match(line)
            if rel:
                src, rtype, raw_props, dst = rel.groups()
                if src not in scope or dst not in scope:
                    dropped += 1                 # the unbound-variable quirk
                    continue
                props = _props(raw_props) if raw_props else {}
                ident = (scope[src], rtype, scope[dst])
                if ident in seen_rels:
                    continue
                seen_rels.add(ident)
                rels.append({
                    "start": scope[src][1], "start_label": scope[src][0],
                    "type": rtype,
                    "end": scope[dst][1], "end_label": scope[dst][0],
                    "props": props,
                })

    return {
        "source": "Neo4j Movies example graph (:play movies)",
        "nodes": sorted(nodes.values(), key=lambda n: (n["label"], n.get("title", n.get("name")))),
        "rels": rels,
        "dropped_unbound_edges": dropped,
    }


def write_json(data: dict, path: Path = MOVIES_JSON) -> Path:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path = MOVIES_JSON) -> dict:
    if not path.exists():                       # fresh clone without the JSON
        write_json(parse_cypher())
    return json.loads(path.read_text(encoding="utf-8"))


# --- in-memory graph -----------------------------------------------------------

@dataclass
class Edge:
    person: str
    type: str
    movie: str | None = None        # set for Person -> Movie edges
    target: str | None = None       # set for Person -> Person (FOLLOWS)
    props: dict = field(default_factory=dict)


class MovieGraph:
    """The Movies graph as Python dicts with an adjacency index both ways.

    The graph is bipartite apart from FOLLOWS (Person -> Person), which is how
    every query in :mod:`kg_rag.templates` can be expressed as one or two
    dictionary lookups.
    """

    def __init__(self, data: dict | None = None):
        data = data or load_json()
        self.movies: dict[str, dict] = {}
        self.people: dict[str, dict] = {}
        for node in data["nodes"]:
            if node["label"] == "Movie":
                self.movies[node["title"]] = node
            else:
                self.people[node["name"]] = node
        self.by_person: dict[str, list[Edge]] = defaultdict(list)
        self.by_movie: dict[str, list[Edge]] = defaultdict(list)
        self.follows: dict[str, list[str]] = defaultdict(list)
        self.rel_counts: dict[str, int] = defaultdict(int)
        for rel in data["rels"]:
            self.rel_counts[rel["type"]] += 1
            if rel["end_label"] == "Movie":
                edge = Edge(rel["start"], rel["type"], movie=rel["end"], props=rel["props"])
                self.by_person[rel["start"]].append(edge)
                self.by_movie[rel["end"]].append(edge)
            else:
                edge = Edge(rel["start"], rel["type"], target=rel["end"], props=rel["props"])
                self.by_person[rel["start"]].append(edge)
                self.follows[rel["start"]].append(rel["end"])

    # -- small helpers used by the template twins --------------------------
    def people_of(self, movie: str, *types: str) -> list[Edge]:
        return [e for e in self.by_movie.get(movie, []) if e.type in types]

    def movies_of(self, person: str, *types: str) -> list[Edge]:
        return [e for e in self.by_person.get(person, []) if e.movie and e.type in types]

    def neighbours(self, name: str) -> list[tuple[str, str]]:
        """Undirected neighbours as (name, rel type) - used for shortest paths."""
        out: list[tuple[str, str]] = []
        if name in self.people:
            for e in self.by_person.get(name, []):
                out.append((e.movie or e.target, e.type))
            for src, targets in self.follows.items():
                if name in targets:
                    out.append((src, "FOLLOWS"))
        elif name in self.movies:
            for e in self.by_movie.get(name, []):
                out.append((e.person, e.type))
        return out

    def stats(self) -> dict:
        return {
            "movies": len(self.movies),
            "people": len(self.people),
            "relationships": sum(self.rel_counts.values()),
            "by_type": dict(sorted(self.rel_counts.items())),
        }
