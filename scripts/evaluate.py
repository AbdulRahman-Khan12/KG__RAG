"""Run both gold sets and write outputs/evaluation.json and outputs/evaluation.csv.

    python scripts/evaluate.py

Uses Neo4j and the LLM if they are configured, otherwise the offline graph and
template answers - the output says which.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_rag.config import DATA_DIR, OUTPUT_DIR  # noqa: E402
from kg_rag.evaluate import evaluate, load_gold  # noqa: E402
from kg_rag.pipeline import MovieRAG  # noqa: E402

rag = MovieRAG()
print(f"backend: {rag.backend.name} - {rag.backend.detail}")
print(f"generator: {'llm' if rag.use_llm else 'template'}\n")

results, rows = {}, []
for name, path in (("gold", DATA_DIR / "eval" / "questions.json"),
                   ("stress", DATA_DIR / "eval" / "stress.json")):
    report = evaluate(rag, load_gold(path))
    results[name] = report.summary()
    print(f"{name} set ({len(report.cases)} questions)")
    for key, value in report.summary().items():
        if key not in ("questions", "generator"):
            print(f"  {key:<18} {value:.3f}")
    for c in report.cases:
        rows.append({"set": name, "question": c.question, "expected_template": c.expected,
                     "predicted_template": c.predicted, "routing_ok": c.template_ok,
                     "slots_ok": c.params_ok, "retrieval_ok": c.retrieval_ok,
                     "answer_ok": c.answer_ok, "grounded": c.grounded, "answer": c.answer})
        if not c.ok:
            print(f"    x {c.question}  (expected {c.expected}, got {c.predicted})")
    print()

results["backend"] = rag.backend.name
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
with (OUTPUT_DIR / "evaluation.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("wrote outputs/evaluation.json and outputs/evaluation.csv")
