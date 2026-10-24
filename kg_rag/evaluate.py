"""Evaluation harness - scores each stage of the RAG loop separately.

A single "accuracy" would hide where a failure happens, so each gold question
is scored on four independent checks:

* **routing**   - the right template was chosen (or none, for out-of-graph questions);
* **linking**   - the template's slots were filled with the right entities;
* **retrieval** - every expected value appears in the rows the query returned;
* **answer**    - every expected value appears in the generated answer, and
  (for LLM answers) no movie or person is mentioned that the rows do not contain.

Declines are scored too: a question about a movie outside the graph must be
refused, not answered with a near miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import GOLD_FILE
from .linker import fold
from .pipeline import MovieRAG


@dataclass
class Case:
    question: str
    template_ok: bool
    params_ok: bool
    retrieval_ok: bool
    answer_ok: bool
    grounded: bool
    predicted: str | None
    expected: str | None
    answer: str
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all([self.template_ok, self.params_ok, self.retrieval_ok, self.answer_ok, self.grounded])


@dataclass
class Report:
    cases: list[Case]
    generator: str

    def rate(self, attr: str) -> float:
        return sum(getattr(c, attr) for c in self.cases) / len(self.cases)

    def summary(self) -> dict:
        return {
            "questions": len(self.cases),
            "generator": self.generator,
            "routing_accuracy": round(self.rate("template_ok"), 3),
            "slot_accuracy": round(self.rate("params_ok"), 3),
            "retrieval_recall": round(self.rate("retrieval_ok"), 3),
            "answer_recall": round(self.rate("answer_ok"), 3),
            "grounded_rate": round(self.rate("grounded"), 3),
            "end_to_end": round(self.rate("ok"), 3),
        }


def load_gold(path=GOLD_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def evaluate(rag: MovieRAG | None = None, gold: list[dict] | None = None) -> Report:
    rag = rag or MovieRAG()
    gold = gold or load_gold()
    cases = []
    for item in gold:
        if not item.get("followup"):
            rag.reset()
        res = rag.ask(item["q"])
        expected_t = item["template"]
        template_ok = res.route.template == expected_t
        params_ok = template_ok and (expected_t is None or res.params == item["params"])
        rows_text = fold(json.dumps(res.rows, default=str, ensure_ascii=False))
        ans_text = fold(res.answer)
        missing = [e for e in item["expect"] if fold(e) not in rows_text or fold(e) not in ans_text]
        retrieval_ok = all(fold(e) in rows_text for e in item["expect"])
        if item.get("empty"):
            retrieval_ok = retrieval_ok and not res.rows
        if "hops" in item:
            retrieval_ok = retrieval_ok and bool(res.rows) and res.rows[0]["hops"] == item["hops"]
        answer_ok = all(fold(e) in ans_text for e in item["expect"])
        if expected_t is None:      # a decline is only correct if nothing was looked up
            retrieval_ok = not res.rows
            answer_ok = res.generator == "none" or "no answer" in ans_text or "couldn't" in ans_text
        cases.append(Case(item["q"], template_ok, params_ok, retrieval_ok, answer_ok,
                          not res.ungrounded, res.route.template, expected_t, res.answer, missing))
    return Report(cases, "llm" if rag.use_llm else "template")
