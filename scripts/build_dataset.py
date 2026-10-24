"""Parse Neo4j's Movies Cypher script into data/movies/movies.json.

    python scripts/build_dataset.py

The JSON is committed, so this only needs re-running if movies.cypher changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_rag.dataset import MovieGraph, parse_cypher, write_json  # noqa: E402

data = parse_cypher()
path = write_json(data)
stats = MovieGraph(data).stats()
print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
print(f"  {stats['movies']} movies, {stats['people']} people, {stats['relationships']} relationships")
for rtype, n in stats["by_type"].items():
    print(f"    {rtype:<10} {n}")
print(f"  unbound edges dropped: {data['dropped_unbound_edges']}")
