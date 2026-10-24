"""The RAG loop: link -> route -> retrieve (Cypher) -> generate -> check.

:class:`MovieRAG` is the only object the interface and the scripts talk to.
It also keeps a one-turn memory so follow-ups work:

    Q: Who directed The Matrix?
    Q: And who acted in it?        <- "it" resolves to The Matrix

The memory holds the last movie and the last person *mentioned in a question*,
not the people in the answer - after "Who directed The Matrix?", "it" should
still mean the movie, not Lilly Wachowski.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import llm, text2cypher
from .backend import BackendError, LocalBackend, connect
from .generator import grounding_check, llm_answer, template_answer
from .linker import EntityLinker, Linked, fold
from .router import Route, route
from .templates import TEMPLATES

MOVIE_PRONOUN = re.compile(r"\b(it|that movie|that film|this movie|this film|the movie|the film)\b")
PERSON_PRONOUN = re.compile(r"\b(he|she|him|her|his|they|them|that actor|that actress|this person)\b")


@dataclass
class Result:
    question: str
    linked: Linked
    route: Route
    cypher: str | None = None
    params: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    answer: str = ""
    generator: str = "template"       # template | llm | none
    backend: str = "offline"
    ungrounded: list[str] = field(default_factory=list)
    context_used: str | None = None
    error: str | None = None
    ms: float = 0.0


class MovieRAG:
    def __init__(self, backend=None, use_llm: bool | None = None):
        self.backend = backend or connect()[0]
        graph = self.backend.graph if isinstance(self.backend, LocalBackend) else LocalBackend().graph
        self.graph = graph          # used for linking and grounding checks only
        self.linker = EntityLinker(graph)
        self.use_llm = llm.available() if use_llm is None else use_llm
        self.last_movie: str | None = None
        self.last_person: str | None = None

    def reset(self) -> None:
        self.last_movie = self.last_person = None

    # -- dialogue context -------------------------------------------------------
    def _fill_from_context(self, question: str, linked: Linked) -> str | None:
        q = fold(question)
        used = None
        if not linked.movies and self.last_movie and MOVIE_PRONOUN.search(q):
            linked.movies.append(self.last_movie)
            used = f"'{MOVIE_PRONOUN.search(q).group(0)}' -> {self.last_movie}"
        if not linked.people and self.last_person and PERSON_PRONOUN.search(q):
            linked.people.append(self.last_person)
            used = f"'{PERSON_PRONOUN.search(q).group(0)}' -> {self.last_person}"
        return used

    # -- the loop -----------------------------------------------------------------
    def ask(self, question: str) -> Result:
        t0 = time.perf_counter()
        linked = self.linker.link(question)
        r = route(question, linked)
        context = None
        if r.template is None:            # nothing to look up - is it a follow-up?
            context = self._fill_from_context(question, linked)
            if context:
                r = route(question, linked)
        result = Result(question, linked, r, backend=self.backend.name, context_used=context)

        if r.template:
            template = TEMPLATES[r.template]
            result.cypher, result.params = template.cypher, r.params
            try:
                result.rows = self.backend.run_template(template, r.params)
            except Exception as exc:
                result.error = f"Query failed: {exc}"
        elif self.use_llm and self.backend.is_neo4j:
            try:
                result.cypher = text2cypher.generate(question)
                result.rows = self.backend.run_cypher(result.cypher)
                result.route = Route(None, {}, "no template matched - LLM wrote the Cypher")
            except Exception as exc:
                result.error = f"Text-to-Cypher failed: {exc}"
        else:
            hint = (" Connect Neo4j and an LLM to let the model write Cypher for "
                    "questions outside the templates." if not self.backend.is_neo4j or not self.use_llm else "")
            result.answer = ("I couldn't find a movie, person or year from the Movies graph in "
                             "that question, so there is nothing to look up." + hint)
            result.generator = "none"

        if not result.answer and not result.error:
            if self.use_llm:
                try:
                    result.answer = llm_answer(question, result.cypher or "", result.rows)
                    result.generator = "llm"
                    result.ungrounded = grounding_check(result.answer, result.rows, self.graph, question)
                except Exception as exc:          # fall back rather than fail
                    result.error = f"LLM unavailable ({exc}); used the template answer."
            if not result.answer:
                if r.template:
                    result.answer = template_answer(r.template, r.params, result.rows)
                    result.generator = "template"
                else:
                    result.answer = f"The query returned {len(result.rows)} rows."
        elif result.error and not result.answer:
            result.answer = "Something went wrong while querying the graph - see the details below."
            result.generator = "none"

        # remember the subject of this question for the next one
        if linked.movies:
            self.last_movie = linked.movies[0]
        if linked.people:
            self.last_person = linked.people[0]
        result.ms = (time.perf_counter() - t0) * 1000
        return result

    def run_cypher(self, query: str) -> list[dict]:
        """Free-form, sanitised Cypher from the Query lab tab (Neo4j only)."""
        if not self.backend.is_neo4j:
            raise BackendError("Free-form Cypher needs a Neo4j database - set NEO4J_URI.")
        return self.backend.run_cypher(text2cypher.sanitise(query))
