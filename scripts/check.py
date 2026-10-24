"""Self-check for KG__RAG - proves every stage works before you demo it.

    python scripts/check.py

Needs no database and no API key. Exits non-zero if anything fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_rag.backend import LocalBackend  # noqa: E402
from kg_rag.dataset import MovieGraph, parse_cypher  # noqa: E402
from kg_rag.evaluate import evaluate  # noqa: E402
from kg_rag.linker import EntityLinker  # noqa: E402
from kg_rag.pipeline import MovieRAG  # noqa: E402
from kg_rag.templates import SAMPLE_PARAMS, TEMPLATES  # noqa: E402
from kg_rag.text2cypher import UnsafeCypher, sanitise  # noqa: E402

RULE = "-" * 64
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print(RULE + "\n1. DATASET\n" + RULE)
data = parse_cypher()
g = MovieGraph(data)
s = g.stats()
check("38 movies", s["movies"] == 38, str(s["movies"]))
check("133 people", s["people"] == 133, str(s["people"]))
check("253 relationships", s["relationships"] == 253, str(s["relationships"]))
check("escaped quotes parsed", "Jerry O'Connell" in g.people)
check("roles kept as lists", g.people_of("The Matrix", "ACTED_IN")[0].props["roles"] != [])

print(RULE + "\n2. ENTITY LINKING\n" + RULE)
linker = EntityLinker(g)
check("longest title wins", linker.link("who directed the matrix reloaded").movies == ["The Matrix Reloaded"])
check("article-less alias", linker.link("cast of matrix").movies == ["The Matrix"])
check("unique surname alias", linker.link("what did hanks act in").people == ["Tom Hanks"])
check("ambiguous surname refused", linker.link("what did thompson review").people == [])
check("typo tolerated", linker.link("who starred in cloud atlass").movies == ["Cloud Atlas"])
check("curly apostrophe", linker.link("You\u2019ve Got Mail").movies == ["You've Got Mail"])
check("years found", linker.link("movies between 1990 and 1995").years == [1990, 1995])

print(RULE + "\n3. EVERY TEMPLATE RETURNS ROWS\n" + RULE)
backend = LocalBackend(g)
for tid, template in TEMPLATES.items():
    rows = backend.run_template(template, SAMPLE_PARAMS[tid])
    check(tid, bool(rows), f"{len(rows)} rows")
    check(f"{tid}: parameterised", all(f"${slot}" in template.cypher for slot in template.slots))

print(RULE + "\n4. CYPHER SAFETY\n" + RULE)
for bad in ["MATCH (n) DETACH DELETE n", "CREATE (:Movie {title:'x'})",
            "MATCH (n) RETURN n; MATCH (m) DELETE m", "CALL dbms.components()",
            "MATCH (n) SET n.x = 1 RETURN n"]:
    try:
        sanitise(bad)
        check(f"blocks: {bad}", False)
    except UnsafeCypher:
        check(f"blocks: {bad}", True)
check("adds a LIMIT", sanitise("MATCH (n) RETURN n").endswith("LIMIT 25"))

print(RULE + "\n5. DIALOGUE MEMORY\n" + RULE)
rag = MovieRAG(backend, use_llm=False)
rag.ask("Who acted in The Matrix?")
follow = rag.ask("Who directed it?")
check("'it' resolves to the last movie", follow.params == {"movie": "The Matrix"})
check("follow-up answered", "Wachowski" in follow.answer)
check("out-of-graph question declined", rag.ask("Who directed Inception?").route.template is None)

print(RULE + "\n6. GOLD SET\n" + RULE)
report = evaluate(MovieRAG(backend, use_llm=False))
for key, value in report.summary().items():
    print(f"  {key:<18} {value}")
check("gold set end-to-end >= 0.95", report.summary()["end_to_end"] >= 0.95)

print(RULE)
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
