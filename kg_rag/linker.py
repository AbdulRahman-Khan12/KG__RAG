"""Entity linking - find the movies, people and years a question mentions.

The graph is small and closed (38 movies, 133 people), so linking is a lookup,
not a model: every title and name is indexed under a few aliases, and the
question is scanned for the longest alias spans first, so "The Matrix
Reloaded" is never mistaken for "The Matrix".

Aliases per entity:

* the exact title or name, case-folded, curly apostrophes straightened;
* a title without its leading article ("matrix" -> The Matrix);
* a person's surname, *only* if no other person shares it ("hanks" works,
  "thompson" does not - three Thompsons review movies).

Typos are handled by a fuzzy pass (difflib ratio >= ``FUZZY_CUTOFF``) over
word n-grams that the exact pass did not consume, restricted to aliases of
six characters or more so that short words cannot fuzzily turn into names.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import FUZZY_CUTOFF
from .dataset import MovieGraph

WORD = re.compile(r"[a-z0-9][a-z0-9'/\-]*")
YEAR = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")


def fold(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").casefold()


@dataclass
class Mention:
    kind: str          # "movie" | "person"
    value: str         # canonical title / name
    surface: str       # what the question actually said
    start: int         # token offset, used to keep mentions in question order
    fuzzy: bool = False


@dataclass
class Linked:
    movies: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.movies or self.people or self.years)


class EntityLinker:
    def __init__(self, graph: MovieGraph):
        self.aliases: dict[tuple[str, ...], tuple[str, str]] = {}
        surnames = Counter(name.split()[-1].casefold() for name in graph.people)
        for title in graph.movies:
            toks = tuple(WORD.findall(fold(title)))
            self.aliases[toks] = ("movie", title)
            if toks and toks[0] in {"the", "a"} and len(toks) > 1:
                self.aliases.setdefault(toks[1:], ("movie", title))
        for name in graph.people:
            toks = tuple(WORD.findall(fold(name)))
            self.aliases[toks] = ("person", name)
            last = toks[-1]
            if surnames[last] == 1 and len(last) > 3:
                self.aliases.setdefault((last,), ("person", name))
        self.max_len = max(len(k) for k in self.aliases)
        self._fuzzy_keys = {" ".join(k): v for k, v in self.aliases.items()
                            if len(" ".join(k)) >= 6}

    def link(self, question: str) -> Linked:
        tokens = WORD.findall(fold(question))
        used = [False] * len(tokens)
        mentions: list[Mention] = []

        def claim(i: int, n: int, kind: str, value: str, fuzzy: bool) -> None:
            for j in range(i, i + n):
                used[j] = True
            mentions.append(Mention(kind, value, " ".join(tokens[i:i + n]), i, fuzzy))

        # exact pass - longest span first
        for n in range(min(self.max_len, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(used[i:i + n]):
                    continue
                hit = self.aliases.get(tuple(tokens[i:i + n]))
                if hit:
                    claim(i, n, *hit, fuzzy=False)

        # fuzzy pass - only over untouched words
        for n in range(min(5, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(used[i:i + n]):
                    continue
                span = " ".join(tokens[i:i + n])
                if len(span) < 6:
                    continue
                best = difflib.get_close_matches(span, self._fuzzy_keys, n=1, cutoff=FUZZY_CUTOFF)
                if best:
                    claim(i, n, *self._fuzzy_keys[best[0]], fuzzy=True)

        mentions.sort(key=lambda m: m.start)
        linked = Linked(mentions=mentions)
        for m in mentions:
            bucket = linked.movies if m.kind == "movie" else linked.people
            if m.value not in bucket:
                bucket.append(m.value)
        # years that are not part of a title ("Apollo 13" has no year, but be safe)
        linked.years = [int(y) for y in YEAR.findall(question)]
        return linked
