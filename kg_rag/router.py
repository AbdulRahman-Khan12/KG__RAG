"""Question -> (template, parameters).

Routing is two decisions made separately:

1. **What is it about?** - the entity linker says which movies, people and
   years the question mentions. These fill the template's slots.
2. **What does it ask?** - keyword cues pick the template among those whose
   slots can be filled.

Keeping them apart is what makes failures diagnosable: a wrong answer is either
a linking error (wrong entity) or a routing error (right entity, wrong
question), and the interface shows both.

A rule fires only if its slots are available, so "who directed it" with no
movie cannot reach ``director_of_movie``; the dialogue layer first tries to
fill the gap from the previous turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .linker import Linked, fold


@dataclass
class Route:
    template: str | None
    params: dict
    reason: str


def _has(q: str, *cues: str) -> bool:
    return any(re.search(rf"\b{cue}", q) for cue in cues)


DIRECT = ("direct",)
CAST = ("act", "star", "cast", "appear", "who was in", "who is in", "who were in", "in it",
        "played in", "featur")
CREW = ("wr[io]t", "screenplay", "script", "produc")
REVIEW = ("review", "rating", "rated", "critic", "think of", "thought of", "opinion")
RECOMMEND = ("recommend", "similar", "suggest", "like ", "should i watch", "fans?\\b",
             "watch next", "if i liked", "if i like")
CONNECT = ("connect", "path", "degree", "link", "separat", "relat", "bacon number")
ROLE = ("role", "play", "character", "who did")


def route(question: str, linked: Linked) -> Route:
    q = fold(question)
    movies, people, years = linked.movies, linked.people, linked.years
    movie = movies[0] if movies else None
    person = people[0] if people else None

    # --- questions about the whole graph (no entity needed) -------------------
    if _has(q, "acted in and directed", "both act", "acted and directed",
            "also directed", "directed and acted", "act in .* they .*direct"):
        return Route("actor_directors", {}, "cue: acted and directed")
    if _has(q, "most movies", "most films", "prolific", "top actor", "busiest", "most roles"):
        return Route("top_actors", {}, "cue: most / prolific")

    # --- two people ------------------------------------------------------------
    if len(people) >= 2 and people[0] != people[1]:
        params = {"a": people[0], "b": people[1]}
        if _has(q, *CONNECT):
            return Route("shortest_path", params, "two people + connection cue")
        return Route("worked_together", params, "two people")

    # --- person + movie --------------------------------------------------------
    if person and movie:
        # the only person+movie template; a role cue just makes the reason clearer
        reason = "person + movie + role cue" if _has(q, *ROLE) else "person + movie"
        return Route("role_in_movie", {"person": person, "movie": movie}, reason)

    # --- one movie ---------------------------------------------------------------
    if movie:
        p = {"movie": movie}
        if _has(q, *REVIEW):
            return Route("reviews_of_movie", p, "movie + review cue")
        if _has(q, *RECOMMEND):
            return Route("recommend_similar", p, "movie + recommendation cue")
        if _has(q, *DIRECT):
            return Route("director_of_movie", p, "movie + direct cue")
        if _has(q, *CREW):
            return Route("crew_of_movie", p, "movie + writer/producer cue")
        if _has(q, *CAST):
            return Route("cast_of_movie", p, "movie + cast cue")
        return Route("movie_details", p, "movie (default: overview)")

    # --- one person ----------------------------------------------------------------
    if person:
        p = {"person": person}
        if _has(q, *RECOMMEND):
            return Route("recommend_for_person", p, "person + recommendation cue")
        if _has(q, "with", "co-?star", "alongside", "colleague", "worked"):
            return Route("co_actors", p, "person + co-star cue")
        if _has(q, *DIRECT):
            return Route("movies_by_director", p, "person + direct cue")
        if _has(q, "born", "birth", "age", "how old"):
            return Route("person_details", p, "person + birth cue")
        if _has(q, *CAST, "movie", "film", "filmography"):
            return Route("movies_of_actor", p, "person + acting cue")
        return Route("person_details", p, "person (default: overview)")

    # --- years only --------------------------------------------------------------------
    if years:
        if len(years) >= 2:
            lo, hi = sorted(years[:2])
            return Route("movies_in_years", {"start": lo, "end": hi}, "two years: range")
        y = years[0]
        if _has(q, "before", "earlier than", "prior to"):
            return Route("movies_in_years", {"start": 0, "end": y - 1}, "year + before")
        if _has(q, "after", "since", "later than"):
            return Route("movies_in_years", {"start": y + 1, "end": 9999}, "year + after")
        return Route("movies_in_years", {"start": y, "end": y}, "single year")

    return Route(None, {}, "no movie, person or year from the graph was found")
