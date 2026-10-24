"""Load the Movies graph into Neo4j, then (optionally) verify every template.

    python scripts/load_neo4j.py            # load (idempotent - safe to re-run)
    python scripts/load_neo4j.py --verify   # load, then compare Neo4j vs offline
    python scripts/load_neo4j.py --reset    # wipe the database first

Connection settings come from .env (see .env.example). Works with Neo4j Desktop
(bolt://localhost:7687) and Neo4j AuraDB Free (neo4j+s://<id>.databases.neo4j.io).

Loading uses UNWIND + MERGE over the parsed JSON rather than replaying the
original script, so it is idempotent: running it twice does not duplicate a
single node.

--verify is the proof that the offline twin and the Cypher agree: every
template is run on both backends with sample parameters and the rows compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_rag import config  # noqa: E402
from kg_rag.backend import LocalBackend, Neo4jBackend  # noqa: E402
from kg_rag.dataset import load_json  # noqa: E402
from kg_rag.templates import SAMPLE_PARAMS, TEMPLATES  # noqa: E402

CONSTRAINTS = [
    "CREATE CONSTRAINT movie_title IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
]
LOAD_MOVIES = """UNWIND $rows AS row
MERGE (m:Movie {title: row.title})
SET m.released = row.released, m.tagline = row.tagline"""
LOAD_PEOPLE = """UNWIND $rows AS row
MERGE (p:Person {name: row.name})
SET p.born = row.born"""
# Relationship types cannot be parameters in Cypher, so one query per type.
LOAD_RELS = """UNWIND $rows AS row
MATCH (a:Person {{name: row.start}})
MATCH (b:{end_label} {{{end_key}: row.end}})
MERGE (a)-[r:{rtype}]->(b)
SET r += row.props"""

def load(db: Neo4jBackend, reset: bool) -> None:
    data = load_json()
    with db.driver.session(database=db.database) as s:
        if reset:
            s.run("MATCH (n) DETACH DELETE n").consume()
            print("  database wiped")
        for c in CONSTRAINTS:
            s.run(c).consume()
        movies = [n for n in data["nodes"] if n["label"] == "Movie"]
        people = [n for n in data["nodes"] if n["label"] == "Person"]
        s.run(LOAD_MOVIES, rows=[{"title": m["title"], "released": m.get("released"),
                                  "tagline": m.get("tagline")} for m in movies]).consume()
        s.run(LOAD_PEOPLE, rows=[{"name": p["name"], "born": p.get("born")} for p in people]).consume()
        by_type: dict[tuple[str, str], list] = {}
        for r in data["rels"]:
            by_type.setdefault((r["type"], r["end_label"]), []).append(
                {"start": r["start"], "end": r["end"], "props": r["props"]})
        for (rtype, end_label), rows in sorted(by_type.items()):
            key = "title" if end_label == "Movie" else "name"
            s.run(LOAD_RELS.format(end_label=end_label, end_key=key, rtype=rtype), rows=rows).consume()
            print(f"  {rtype:<10} {len(rows):>4} relationships")
    print("  loaded:", db.stats())


def verify(db: Neo4jBackend) -> int:
    local = LocalBackend()
    failures = 0
    for tid, template in TEMPLATES.items():
        params = SAMPLE_PARAMS[tid]
        remote_rows = db.run_template(template, params)
        local_rows = local.run_template(template, params)
        if tid == "shortest_path":          # several shortest paths can exist
            same = [r["hops"] for r in remote_rows] == [r["hops"] for r in local_rows]
        else:
            same = remote_rows == local_rows
        failures += not same
        print(f"  [{'PASS' if same else 'FAIL'}] {tid:<22} neo4j={len(remote_rows):>2} rows  offline={len(local_rows):>2} rows")
        if not same:
            print("      neo4j  :", remote_rows[:3])
            print("      offline:", local_rows[:3])
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if not config.NEO4J_URI:
        print("NEO4J_URI is not set. Copy .env.example to .env and fill it in.")
        return 1
    print(f"Connecting to {config.NEO4J_URI} ...")
    db = Neo4jBackend(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    load(db, args.reset)
    failures = verify(db) if args.verify else 0
    db.close()
    if args.verify:
        print("all templates agree" if not failures else f"{failures} template(s) disagree")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
