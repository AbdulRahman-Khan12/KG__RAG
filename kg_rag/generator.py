"""Answer generation - the "G" in RAG.

Two generators share one contract: *the answer may only state what the
retrieved rows contain.*

* :func:`template_answer` verbalises rows with a sentence pattern per template.
  It needs no API key, is deterministic, and is grounded by construction.
* :func:`llm_answer` hands the question, the Cypher that ran and the rows (as
  JSON) to an LLM with an instruction to answer from the rows alone. It reads
  more naturally and copes with phrasing the patterns do not anticipate, but it
  *can* hallucinate - so every LLM answer is passed through
  :func:`grounding_check`.

The grounding check is cheap because the graph is closed: any movie title or
person name the answer mentions must also appear in the retrieved rows. A name
that appears in the answer but not in the evidence is reported as ungrounded.
"""

from __future__ import annotations

import json
import re

from . import llm
from .dataset import MovieGraph
from .linker import fold


def _join(items: list[str], limit: int = 8) -> str:
    items = [str(i) for i in items]
    if len(items) > limit:
        return ", ".join(items[:limit]) + f" and {len(items) - limit} more"
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _roles(roles: list[str]) -> str:
    return " / ".join(roles) if roles else "an uncredited role"


def template_answer(template_id: str, params: dict, rows: list[dict]) -> str:
    p = params
    if not rows:
        return _empty(template_id, p)
    r0 = rows[0]
    if template_id == "cast_of_movie":
        return (f"{p['movie']} starred " +
                _join([f"{r['actor']} ({_roles(r['roles'])})" for r in rows]) + ".")
    if template_id == "movies_of_actor":
        return (f"{p['person']} acted in {len(rows)} movie{'s' * (len(rows) != 1)} in the graph: " +
                _join([f"{r['title']} ({r['released']})" for r in rows], 12) + ".")
    if template_id == "director_of_movie":
        return f"{p['movie']} was directed by {_join([r['director'] for r in rows])}."
    if template_id == "movies_by_director":
        return (f"{p['person']} directed " +
                _join([f"{r['title']} ({r['released']})" for r in rows]) + ".")
    if template_id == "crew_of_movie":
        wrote = [r["person"] for r in rows if r["role"] == "WROTE"]
        produced = [r["person"] for r in rows if r["role"] == "PRODUCED"]
        if wrote and sorted(wrote) == sorted(produced):
            return f"{p['movie']} was written and produced by {_join(wrote)}."
        parts = []
        if wrote:
            parts.append(f"written by {_join(wrote)}")
        if produced:
            parts.append(f"produced by {_join(produced)}")
        return f"{p['movie']} was " + " and ".join(parts) + "."
    if template_id == "movie_details":
        text = f"{r0['title']} was released in {r0['released']}"
        if r0.get("directors"):
            text += f" and directed by {_join(r0['directors'])}"
        text += "."
        if r0.get("tagline"):
            tagline = r0["tagline"]
            text += f" Tagline: “{tagline}”" + ("" if tagline[-1] in ".!?" else ".")
        if r0.get("cast"):
            more = r0["cast_size"] - len(r0["cast"])
            text += f" Cast includes {_join(r0['cast'])}" + (f" (+{more} more)" if more > 0 else "") + "."
        return text
    if template_id == "person_details":
        credits = [f"{n} {label}" for label, n in
                   (("acting", r0["acted"]), ("directing", r0["directed"]),
                    ("producing", r0["produced"]), ("writing", r0["wrote"])) if n]
        born = f"was born in {r0['born']}" if r0.get("born") else "has no birth year in the graph"
        work = f" and has {_join(credits)} credit{'s' if sum([r0['acted'], r0['directed'], r0['produced'], r0['wrote']]) != 1 else ''}" if credits else ""
        return f"{r0['name']} {born}{work}."
    if template_id == "role_in_movie":
        return f"In {r0['title']}, {r0['actor']} played {_roles(r0['roles'])}."
    if template_id == "co_actors":
        top = [f"{r['co_actor']} ({r['movies_together']})" for r in rows[:10]]
        return (f"{p['person']} has acted alongside {len(rows)} people in the graph. "
                f"Most frequent: {_join(top, 10)} - number of shared movies in brackets.")
    if template_id == "shortest_path":
        chain = r0["nodes"][0]
        for rel, node in zip(r0["rels"], r0["nodes"][1:]):
            chain += f" —{rel}— {node}"
        return f"{p['a']} and {p['b']} are {r0['hops']} hops apart: {chain}."
    if template_id == "movies_in_years":
        span = (f"in {p['start']}" if p["start"] == p["end"] else
                f"before {p['end'] + 1}" if p["start"] == 0 else
                f"after {p['start'] - 1}" if p["end"] == 9999 else
                f"between {p['start']} and {p['end']}")
        return (f"{len(rows)} movie{'s' * (len(rows) != 1)} released {span}: " +
                _join([f"{r['title']} ({r['released']})" for r in rows], 15) + ".")
    if template_id == "recommend_similar":
        recs = [f"{r['title']} ({r['released']}; shares {_join(r['shared'], 3)})" for r in rows]
        return f"If you liked {p['movie']}, try: " + "; ".join(recs) + "."
    if template_id == "recommend_for_person":
        recs = [f"{r['title']} ({r['released']}; via {_join(r['via'], 3)})" for r in rows]
        return (f"Movies {p['person']} is not in, reached through co-actors: " +
                "; ".join(recs) + ".")
    if template_id == "reviews_of_movie":
        avg = sum(r["rating"] for r in rows) / len(rows)
        quotes = "; ".join(f"{r['reviewer']} gave it {r['rating']} (“{r['summary']}”)" for r in rows)
        return f"{p['movie']} has {len(rows)} review{'s' * (len(rows) != 1)}, average rating {avg:.0f}/100. {quotes}."
    if template_id == "top_actors":
        return "Most prolific actors in the graph: " + _join([f"{r['actor']} ({r['movies']})" for r in rows], 10) + "."
    if template_id == "actor_directors":
        return "People who acted in a movie they also directed: " + _join(
            [f"{r['person']} ({r['title']})" for r in rows]) + "."
    if template_id == "worked_together":
        label = {"ACTED_IN": "acted", "DIRECTED": "directed", "PRODUCED": "produced", "WROTE": "wrote"}
        items = [f"{r['title']} ({r['released']}; {p['a']} {label[r['a_role']]}, "
                 f"{p['b']} {label[r['b_role']]})" for r in rows]
        return f"{p['a']} and {p['b']} share {len(rows)} credit{'s' * (len(rows) != 1)}: " + "; ".join(items) + "."
    return json.dumps(rows[:5], default=str)


def _empty(template_id: str, p: dict) -> str:
    messages = {
        "role_in_movie": f"The graph has no record of {p.get('person')} acting in {p.get('movie')}.",
        "reviews_of_movie": f"Nobody in the graph has reviewed {p.get('movie')}.",
        "shortest_path": f"No connection within 6 hops between {p.get('a')} and {p.get('b')}.",
        "worked_together": f"{p.get('a')} and {p.get('b')} share no movie in the graph.",
        "director_of_movie": f"The graph does not record a director for {p.get('movie')}.",
        "crew_of_movie": f"The graph records no writer or producer for {p.get('movie')}.",
        "movies_by_director": f"{p.get('person')} has no directing credits in the graph.",
        "movies_of_actor": f"{p.get('person')} has no acting credits in the graph.",
        "co_actors": f"{p.get('person')} has no co-actors in the graph.",
        "movies_in_years": "No movie in the graph was released in that period.",
    }
    return messages.get(template_id, "The graph returned no rows for this question.")


SYSTEM_PROMPT = """You answer questions about movies using ONLY the database rows provided.
Rules:
- Use only facts present in the rows. Do not add outside knowledge, even if you know it.
- If the rows are empty or do not answer the question, say the database has no answer.
- Copy names and titles exactly as they appear in the rows.
- Answer in at most three sentences. No preamble, no markdown headings."""


def llm_answer(question: str, cypher: str, rows: list[dict]) -> str:
    user = (f"Question: {question}\n\nCypher query that was run:\n{cypher}\n\n"
            f"Rows returned ({len(rows)}):\n{json.dumps(rows, default=str, indent=1)}")
    return llm.chat(SYSTEM_PROMPT, user)


def _mentions(text: str, graph: MovieGraph) -> set[str]:
    folded = fold(text)
    found = set()
    for name in list(graph.movies) + list(graph.people):
        if re.search(rf"(?<![a-z0-9]){re.escape(fold(name))}(?![a-z0-9])", folded):
            found.add(name)
    # "The Matrix" is a substring of "The Matrix Reloaded"; keep the longest only
    return {n for n in found if not any(n != m and fold(n) in fold(m) for m in found)}


def grounding_check(answer: str, rows: list[dict], graph: MovieGraph,
                    question: str = "") -> list[str]:
    """Names the answer mentions that are in neither the rows nor the question."""
    evidence = json.dumps(rows, default=str, ensure_ascii=False) + " " + question
    allowed = _mentions(evidence, graph)
    return sorted(_mentions(answer, graph) - allowed)
