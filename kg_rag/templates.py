"""Cypher query templates - the retrieval half of the RAG system.

Each template is one question *shape* ("who directed <movie>?") paired with:

* the parameterised Cypher query that answers it on a real Neo4j database, and
* a Python twin that computes the same rows from the in-memory graph.

The twin exists so the project runs on a fresh clone with no database. The
Cypher is the real deliverable: it is what runs whenever Neo4j is configured,
it is shown to the user next to every answer, and it is listed in
``docs/cypher_templates.md``. Both versions return rows with identical column
names and identical ordering, so the generator never knows which backend
answered. ``scripts/check.py`` runs every template through the offline twin;
``scripts/load_neo4j.py --verify`` runs every Cypher query against Neo4j and
compares the two.

Why templates instead of letting an LLM write Cypher? A template is correct by
construction, needs no API key, can never write to the database, and is
auditable line by line. LLM-written Cypher is kept only as an optional
fallback for questions no template covers (see :mod:`kg_rag.text2cypher`).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

from .config import MAX_RECORDS, PATH_MAX_HOPS, RECOMMEND_LIMIT
from .dataset import MovieGraph

COLLAB = ("ACTED_IN", "DIRECTED", "PRODUCED", "WROTE")


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    example: str
    slots: tuple[str, ...]
    cypher: str
    local: Callable[[MovieGraph, dict], list[dict]] = field(repr=False)
    sort_lists: tuple[str, ...] = ()     # list columns whose order Cypher can't fix


def _rows(rows: list[dict]) -> list[dict]:
    return rows[:MAX_RECORDS]


# --- the twins -----------------------------------------------------------------

def _cast(g, p):
    rows = [{"actor": e.person, "roles": e.props.get("roles", [])}
            for e in g.people_of(p["movie"], "ACTED_IN")]
    return _rows(sorted(rows, key=lambda r: r["actor"]))


def _filmography(g, p):
    rows = [{"title": e.movie, "released": g.movies[e.movie].get("released"),
             "roles": e.props.get("roles", [])} for e in g.movies_of(p["person"], "ACTED_IN")]
    return _rows(sorted(rows, key=lambda r: (r["released"], r["title"])))


def _director_of(g, p):
    return sorted(({"director": e.person} for e in g.people_of(p["movie"], "DIRECTED")),
                  key=lambda r: r["director"])


def _directed_by(g, p):
    rows = [{"title": e.movie, "released": g.movies[e.movie].get("released")}
            for e in g.movies_of(p["person"], "DIRECTED")]
    return _rows(sorted(rows, key=lambda r: (r["released"], r["title"])))


def _crew(g, p):
    rows = [{"person": e.person, "role": e.type}
            for e in g.people_of(p["movie"], "WROTE", "PRODUCED")]
    return sorted(rows, key=lambda r: (r["role"], r["person"]))


def _movie_details(g, p):
    movie = g.movies.get(p["movie"])
    if not movie:
        return []
    cast = sorted(e.person for e in g.people_of(p["movie"], "ACTED_IN"))
    directors = sorted(e.person for e in g.people_of(p["movie"], "DIRECTED"))
    return [{"title": movie["title"], "released": movie.get("released"),
             "tagline": movie.get("tagline"), "directors": directors,
             "cast": cast[:5], "cast_size": len(cast)}]


def _person_details(g, p):
    person = g.people.get(p["person"])
    if not person:
        return []
    counts = defaultdict(int)
    for e in g.movies_of(p["person"], *COLLAB):
        counts[e.type] += 1
    return [{"name": person["name"], "born": person.get("born"),
             "acted": counts["ACTED_IN"], "directed": counts["DIRECTED"],
             "produced": counts["PRODUCED"], "wrote": counts["WROTE"]}]


def _role(g, p):
    return [{"actor": e.person, "title": e.movie, "roles": e.props.get("roles", [])}
            for e in g.movies_of(p["person"], "ACTED_IN") if e.movie == p["movie"]]


def _co_actors(g, p):
    shared = defaultdict(set)
    for e in g.movies_of(p["person"], "ACTED_IN"):
        for other in g.people_of(e.movie, "ACTED_IN"):
            if other.person != p["person"]:
                shared[other.person].add(e.movie)
    rows = [{"co_actor": c, "movies_together": len(m), "shared": sorted(m)}
            for c, m in shared.items()]
    return _rows(sorted(rows, key=lambda r: (-r["movies_together"], r["co_actor"])))


def _shortest_path(g, p):
    """Breadth-first search over collaboration edges, ignoring direction."""
    start, goal = p["a"], p["b"]
    if start not in g.people or goal not in g.people:
        return []
    parent: dict[str, tuple[str, str] | None] = {start: None}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if node == goal:
            break
        if depth >= PATH_MAX_HOPS:
            continue
        for nxt, rtype in sorted(g.neighbours(node)):
            if rtype in COLLAB and nxt not in parent:
                parent[nxt] = (node, rtype)
                queue.append((nxt, depth + 1))
    if goal not in parent:
        return []
    nodes, rels, cur = [goal], [], goal
    while parent[cur] is not None:
        prev, rtype = parent[cur]
        nodes.append(prev)
        rels.append(rtype)
        cur = prev
    nodes.reverse()
    rels.reverse()
    return [{"nodes": nodes, "rels": rels, "hops": len(rels)}]


def _in_years(g, p):
    rows = [{"title": t, "released": m.get("released")} for t, m in g.movies.items()
            if m.get("released") is not None and p["start"] <= m["released"] <= p["end"]]
    return _rows(sorted(rows, key=lambda r: (r["released"], r["title"])))


def _similar(g, p):
    target, shared = p["movie"], defaultdict(set)
    for e in g.people_of(target, "ACTED_IN", "DIRECTED"):
        for other in g.movies_of(e.person, "ACTED_IN", "DIRECTED"):
            if other.movie != target:
                shared[other.movie].add(e.person)
    rows = [{"title": t, "released": g.movies[t].get("released"),
             "score": len(people), "shared": sorted(people)} for t, people in shared.items()]
    rows.sort(key=lambda r: (-r["score"], -(r["released"] or 0), r["title"]))
    return rows[:RECOMMEND_LIMIT]


def _for_person(g, p):
    me = p["person"]
    seen = {e.movie for e in g.movies_of(me, "ACTED_IN")}
    via = defaultdict(set)
    for movie in seen:
        for co in g.people_of(movie, "ACTED_IN"):
            if co.person == me:
                continue
            for rec in g.movies_of(co.person, "ACTED_IN"):
                if rec.movie not in seen:
                    via[rec.movie].add(co.person)
    rows = [{"title": t, "released": g.movies[t].get("released"),
             "score": len(people), "via": sorted(people)} for t, people in via.items()]
    rows.sort(key=lambda r: (-r["score"], -(r["released"] or 0), r["title"]))
    return rows[:RECOMMEND_LIMIT]


def _reviews(g, p):
    rows = [{"reviewer": e.person, "rating": e.props.get("rating"),
             "summary": e.props.get("summary")} for e in g.people_of(p["movie"], "REVIEWED")]
    return sorted(rows, key=lambda r: (-(r["rating"] or 0), r["reviewer"]))


def _top_actors(g, p):
    counts = defaultdict(int)
    for person, edges in g.by_person.items():
        counts[person] += sum(1 for e in edges if e.type == "ACTED_IN")
    rows = [{"actor": a, "movies": n} for a, n in counts.items() if n]
    return sorted(rows, key=lambda r: (-r["movies"], r["actor"]))[:10]


def _actor_directors(g, p):
    rows = []
    for person, edges in g.by_person.items():
        acted = {e.movie for e in edges if e.type == "ACTED_IN"}
        rows += [{"person": person, "title": e.movie}
                 for e in edges if e.type == "DIRECTED" and e.movie in acted]
    return sorted(rows, key=lambda r: (r["person"], r["title"]))


def _together(g, p):
    a_edges = [e for e in g.movies_of(p["a"], *COLLAB)]
    b_edges = [e for e in g.movies_of(p["b"], *COLLAB)]
    rows = [{"title": ea.movie, "released": g.movies[ea.movie].get("released"),
             "a_role": ea.type, "b_role": eb.type}
            for ea in a_edges for eb in b_edges if ea.movie == eb.movie]
    return sorted(rows, key=lambda r: (r["released"], r["title"], r["a_role"], r["b_role"]))


# --- the catalogue ---------------------------------------------------------------

TEMPLATES: dict[str, Template] = {t.id: t for t in [
    Template(
        "cast_of_movie", "Cast of a movie", "Who acted in The Matrix?", ("movie",),
        """MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {title: $movie})
RETURN p.name AS actor, r.roles AS roles
ORDER BY actor
LIMIT 25""", _cast),
    Template(
        "movies_of_actor", "Filmography of an actor", "Which movies did Tom Hanks act in?",
        ("person",),
        """MATCH (p:Person {name: $person})-[r:ACTED_IN]->(m:Movie)
RETURN m.title AS title, m.released AS released, r.roles AS roles
ORDER BY released, title
LIMIT 25""", _filmography),
    Template(
        "director_of_movie", "Director of a movie", "Who directed Cloud Atlas?", ("movie",),
        """MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: $movie})
RETURN p.name AS director
ORDER BY director""", _director_of),
    Template(
        "movies_by_director", "Movies by a director", "What did Ron Howard direct?", ("person",),
        """MATCH (p:Person {name: $person})-[:DIRECTED]->(m:Movie)
RETURN m.title AS title, m.released AS released
ORDER BY released, title
LIMIT 25""", _directed_by),
    Template(
        "crew_of_movie", "Writers and producers", "Who wrote and produced Jerry Maguire?",
        ("movie",),
        """MATCH (p:Person)-[r:WROTE|PRODUCED]->(m:Movie {title: $movie})
RETURN p.name AS person, type(r) AS role
ORDER BY role, person""", _crew),
    Template(
        "movie_details", "Movie overview", "Tell me about Top Gun", ("movie",),
        """MATCH (m:Movie {title: $movie})
OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
WITH m, a ORDER BY a.name
WITH m, collect(a.name) AS full_cast
OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
WITH m, full_cast, d ORDER BY d.name
WITH m, full_cast, collect(d.name) AS directors
RETURN m.title AS title, m.released AS released, m.tagline AS tagline,
       directors, full_cast[..5] AS cast, size(full_cast) AS cast_size""",
        _movie_details),
    Template(
        "person_details", "Person overview", "When was Keanu Reeves born?", ("person",),
        """MATCH (p:Person {name: $person})
OPTIONAL MATCH (p)-[r:ACTED_IN|DIRECTED|PRODUCED|WROTE]->(:Movie)
RETURN p.name AS name, p.born AS born,
       sum(CASE type(r) WHEN 'ACTED_IN' THEN 1 ELSE 0 END) AS acted,
       sum(CASE type(r) WHEN 'DIRECTED' THEN 1 ELSE 0 END) AS directed,
       sum(CASE type(r) WHEN 'PRODUCED' THEN 1 ELSE 0 END) AS produced,
       sum(CASE type(r) WHEN 'WROTE' THEN 1 ELSE 0 END) AS wrote""", _person_details),
    Template(
        "role_in_movie", "Role played in a movie", "Who did Keanu Reeves play in The Matrix?",
        ("person", "movie"),
        """MATCH (p:Person {name: $person})-[r:ACTED_IN]->(m:Movie {title: $movie})
RETURN p.name AS actor, m.title AS title, r.roles AS roles""", _role),
    Template(
        "co_actors", "Co-actors", "Who has acted with Meg Ryan?", ("person",),
        """MATCH (p:Person {name: $person})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(c:Person)
WITH c, collect(m.title) AS shared
RETURN c.name AS co_actor, size(shared) AS movies_together, shared
ORDER BY movies_together DESC, co_actor
LIMIT 25""", _co_actors, sort_lists=("shared",)),
    Template(
        "shortest_path", "How two people are connected",
        "How is Kevin Bacon connected to Tom Cruise?", ("a", "b"),
        f"""MATCH (a:Person {{name: $a}}), (b:Person {{name: $b}})
MATCH path = shortestPath((a)-[:ACTED_IN|DIRECTED|PRODUCED|WROTE*..{PATH_MAX_HOPS}]-(b))
RETURN [n IN nodes(path) | coalesce(n.name, n.title)] AS nodes,
       [r IN relationships(path) | type(r)] AS rels,
       length(path) AS hops""", _shortest_path),
    Template(
        "movies_in_years", "Movies by release year", "Which movies came out in 1999?",
        ("start", "end"),
        """MATCH (m:Movie)
WHERE m.released >= $start AND m.released <= $end
RETURN m.title AS title, m.released AS released
ORDER BY released, title
LIMIT 25""", _in_years),
    Template(
        "recommend_similar", "Recommend similar movies", "Recommend movies like The Matrix",
        ("movie",),
        f"""MATCH (m:Movie {{title: $movie}})<-[:ACTED_IN|DIRECTED]-(p:Person)
      -[:ACTED_IN|DIRECTED]->(rec:Movie)
WHERE rec <> m
WITH rec, collect(DISTINCT p.name) AS shared
RETURN rec.title AS title, rec.released AS released, size(shared) AS score, shared
ORDER BY score DESC, released DESC, title
LIMIT {RECOMMEND_LIMIT}""", _similar, sort_lists=("shared",)),
    Template(
        "recommend_for_person", "Recommend via co-actors",
        "What should a Tom Hanks fan watch next?", ("person",),
        f"""MATCH (p:Person {{name: $person}})-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(c:Person)
      -[:ACTED_IN]->(rec:Movie)
WHERE c <> p AND NOT (p)-[:ACTED_IN]->(rec)
WITH rec, collect(DISTINCT c.name) AS via
RETURN rec.title AS title, rec.released AS released, size(via) AS score, via
ORDER BY score DESC, released DESC, title
LIMIT {RECOMMEND_LIMIT}""", _for_person, sort_lists=("via",)),
    Template(
        "reviews_of_movie", "Reviews and ratings", "What did reviewers think of Cloud Atlas?",
        ("movie",),
        """MATCH (r:Person)-[rev:REVIEWED]->(m:Movie {title: $movie})
RETURN r.name AS reviewer, rev.rating AS rating, rev.summary AS summary
ORDER BY rating DESC, reviewer""", _reviews),
    Template(
        "top_actors", "Most prolific actors", "Who acted in the most movies?", (),
        """MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, count(m) AS movies
ORDER BY movies DESC, actor
LIMIT 10""", _top_actors),
    Template(
        "actor_directors", "Acted in and directed the same movie",
        "Who acted in a movie they also directed?", (),
        """MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(p)
RETURN p.name AS person, m.title AS title
ORDER BY person, title""", _actor_directors),
    Template(
        "worked_together", "Movies two people share",
        "Which movies did Tom Hanks and Meg Ryan make together?", ("a", "b"),
        """MATCH (a:Person {name: $a})-[ra:ACTED_IN|DIRECTED|PRODUCED|WROTE]->(m:Movie)
      <-[rb:ACTED_IN|DIRECTED|PRODUCED|WROTE]-(b:Person {name: $b})
RETURN m.title AS title, m.released AS released, type(ra) AS a_role, type(rb) AS b_role
ORDER BY released, title, a_role, b_role""", _together),
]}


# One known-good parameter set per template - used by the self-check and by
# scripts/load_neo4j.py --verify to compare Neo4j against the offline twin.
SAMPLE_PARAMS = {
    "cast_of_movie": {"movie": "The Matrix"},
    "movies_of_actor": {"person": "Tom Hanks"},
    "director_of_movie": {"movie": "Cloud Atlas"},
    "movies_by_director": {"person": "Ron Howard"},
    "crew_of_movie": {"movie": "Jerry Maguire"},
    "movie_details": {"movie": "Top Gun"},
    "person_details": {"person": "Keanu Reeves"},
    "role_in_movie": {"person": "Keanu Reeves", "movie": "The Matrix"},
    "co_actors": {"person": "Meg Ryan"},
    "shortest_path": {"a": "Kevin Bacon", "b": "Tom Cruise"},
    "movies_in_years": {"start": 1990, "end": 1995},
    "recommend_similar": {"movie": "The Matrix"},
    "recommend_for_person": {"person": "Tom Hanks"},
    "reviews_of_movie": {"movie": "The Replacements"},
    "top_actors": {},
    "actor_directors": {},
    "worked_together": {"a": "Tom Hanks", "b": "Meg Ryan"},
}


def normalise(template: Template, rows: list[dict]) -> list[dict]:
    """Make Neo4j rows and twin rows directly comparable (plain types, sorted lists)."""
    out = []
    for row in rows:
        clean = {}
        for key, value in dict(row).items():
            if isinstance(value, (list, tuple)):
                value = list(value)
                if key in template.sort_lists:
                    value = sorted(value)
            clean[key] = value
        out.append(clean)
    return out
