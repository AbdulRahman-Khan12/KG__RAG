# KG__RAG

Retrieval-augmented question answering over the **Neo4j Movies knowledge graph**.
Questions are turned into parameterised Cypher queries, the graph returns the
facts, and an answer is written from those facts alone.

Built for **AML23702 Advance Natural Language Processing, Lab Exercise 3** —
a small RAG application with a knowledge-graph backend.

**Runs in one command on a fresh clone** — no database, no API key needed.
Neo4j and an LLM are switched on by filling in `.env`.

---

## Architecture

```
question
   │
   ├── entity linker ──── movies / people / years ── (exact + fuzzy, longest span first)
   │
   ├── router ─────────── which of 17 Cypher templates? ── slots filled from the linker
   │        │                                              follow-ups reuse the last subject
   │        └── no template fits ── LLM writes Cypher (optional, read-only, sanitised)
   │
   ├── retriever ──────── Neo4j (bolt / AuraDB)      ── runs the Cypher
   │                      or offline in-memory graph ── runs the same query's Python twin
   │
   └── generator ──────── LLM, prompted with the rows only ── grounding check
                          or template sentence (no key)    ── grounded by construction
```

Every answer in the app is shown with the evidence behind it: the linked
entities, the template and the reason it was chosen, the exact Cypher and
parameters, and the rows the graph returned.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

That is all that is needed: the app starts on the bundled copy of the Movies
graph and answers with template sentences. The sidebar says which backend and
which generator are live.

### Connecting Neo4j (the knowledge graph backend)

Pick one:

| Option | Steps |
|---|---|
| **Neo4j AuraDB Free** (cloud, nothing to install) | Create a free instance at console.neo4j.io, download the credentials file |
| **Neo4j Desktop** | Create a local DBMS, start it, note the password |
| **Docker** | `docker compose up -d` (password `movies-graph`) |

Then:

```bash
cp .env.example .env              # fill in NEO4J_URI and NEO4J_PASSWORD
python scripts/load_neo4j.py --verify
streamlit run app.py
```

`load_neo4j.py` loads the 38 movies, 133 people and 253 relationships with
idempotent `UNWIND … MERGE` queries (safe to run twice), then `--verify` runs
all 17 templates on Neo4j **and** on the offline twin and checks the rows match.

### Connecting an LLM (optional)

Any OpenAI-compatible endpoint: Groq (free tier), OpenAI, Gemini, OpenRouter,
or a local Ollama server with no key at all. Set `LLM_API_KEY`, `LLM_BASE_URL`
and `LLM_MODEL` in `.env` — examples for each provider are in `.env.example`.
With an LLM *and* Neo4j connected, questions that no template covers are
answered by LLM-written Cypher.

## Build stages

| Stage | Scope | Status |
|-------|-------|--------|
| 1 | Movies dataset: Cypher script → JSON, in-memory graph | **done** |
| 2 | 17 parameterised Cypher templates + offline twins | **done** |
| 3 | Entity linking, template routing, follow-up memory | **done** |
| 4 | Generation: template sentences, LLM, grounding check, text-to-Cypher | **done** |
| 5 | Streamlit interface, Neo4j loader, evaluation | **done** |

## Layout

```
KG__RAG/
├── app.py                  the Streamlit interface
├── kg_rag/
│   ├── config.py           every setting, read from .env / the environment
│   ├── dataset.py          parses movies.cypher; MovieGraph (in-memory, both directions)
│   ├── templates.py        17 Cypher templates, each with a Python twin
│   ├── linker.py           entity linking - exact aliases, then fuzzy
│   ├── router.py           question -> (template, parameters)
│   ├── backend.py          Neo4jBackend (READ sessions) and LocalBackend
│   ├── generator.py        template answers, LLM answers, grounding check
│   ├── llm.py              OpenAI-compatible client on urllib - no SDK
│   ├── text2cypher.py      optional LLM-written Cypher, sanitised
│   ├── pipeline.py         MovieRAG: link -> route -> retrieve -> generate
│   └── evaluate.py         routing / slots / retrieval / answer scored separately
├── data/
│   ├── movies/movies.cypher   Neo4j's original Movies script (:play movies)
│   ├── movies/movies.json     the same graph, parsed
│   └── eval/                  questions.json (32 gold), stress.json (15 paraphrases)
├── docs/cypher_templates.md   every template, generated from the code
├── outputs/                   evaluation.json / evaluation.csv
└── scripts/
    ├── check.py               self-check, no database or key needed
    ├── load_neo4j.py          load + verify Neo4j against the offline twin
    ├── evaluate.py            both question sets -> outputs/
    ├── build_dataset.py       movies.cypher -> movies.json
    └── export_templates.py    templates -> docs/cypher_templates.md
```

## Knowledge graph schema

```
(:Person {name, born}) -[:ACTED_IN {roles}]->   (:Movie {title, released, tagline})
(:Person)              -[:DIRECTED]->           (:Movie)
(:Person)              -[:PRODUCED]->           (:Movie)
(:Person)              -[:WROTE]->              (:Movie)
(:Person)              -[:REVIEWED {summary, rating}]-> (:Movie)
(:Person)              -[:FOLLOWS]->            (:Person)
```

38 movies · 133 people · 253 relationships (ACTED_IN 172, DIRECTED 44,
PRODUCED 15, WROTE 10, REVIEWED 9, FOLLOWS 3).

## Results

Run `python scripts/evaluate.py` (offline graph, template answers):

| Measure | Gold (32) | Stress (15) |
|---|---|---|
| Routing accuracy | 1.000 | 0.600 |
| Slot (entity) accuracy | 1.000 | 0.600 |
| Retrieval recall | 1.000 | 0.733 |
| Answer recall | 1.000 | 0.733 |
| Grounded answers | 1.000 | 1.000 |
| **End to end** | **1.000** | **0.600** |

The gold set includes follow-ups ("who directed **it**?"), a typo
("matrx revolutions"), a curly apostrophe, two out-of-graph questions that
must be declined, and one question whose correct answer is "no such credit".

The **stress set was written after the router was finished and never used to
tune it**, so it is the honest measure of generalisation. Its six failures are
all routing gaps, not retrieval errors: an unseen synonym (*helmed*), a vague
time span (*early 2000s*), an ambiguous surname (*Wachowski*), a question no
template can answer (*oldest actor*), and recommendation cues the keyword list
does not contain. These are exactly the questions the optional LLM
text-to-Cypher path exists for.

## Running the checks

```bash
python scripts/check.py      # dataset, linking, all 17 templates, Cypher safety, memory, gold set
python scripts/evaluate.py   # both question sets, written to outputs/
```

## Design notes

**Why templates first.** A template is correct by construction, needs no API
key, cannot write to the database and can be audited line by line. Most movie
questions are one of a few shapes, and 17 templates cover them. LLM-written
Cypher is kept as a fallback, not the main path.

**Why parameters, never string formatting.** User text only ever reaches
Cypher as `$movie` / `$person`, so a question cannot inject a query. The one
exception — relationship types in the loader — comes from the dataset, not
the user.

**Why an offline twin.** The assignment is about the RAG loop, and a demo
that dies because a cloud database is paused teaches nothing. Every template
has a Python twin over the same JSON; `load_neo4j.py --verify` proves the two
agree row for row. The interface always says which backend answered.

**Why a grounding check.** The graph is closed, so any movie or person an
answer names must appear in the retrieved rows. An LLM answer that names
anyone else is flagged as ungrounded in the interface and counted in the
evaluation.

**Why the memory stores the question's subject.** After "Who directed The
Matrix?", "it" must still mean the movie, not the director in the answer.

**Why READ sessions.** Even LLM-written Cypher that slipped past the keyword
filter cannot change the graph: every query runs with `READ_ACCESS`.

## Limitations

- Routing is keyword-based; unseen synonyms fall through (see the stress set).
- Pronoun resolution is one turn deep and gender-blind ("her" resolves to the
  last person, whoever that was).
- Surnames shared by two people (*Wachowski*, *Thompson*) are not linked
  rather than guessed.
- The Movies graph is tiny and ends in 2012; recommendations are by shared
  cast and crew only, since the graph has no genres.

## Data

Movies dataset © Neo4j, from the `:play movies` guide
(github.com/neo4j-graph-examples/movies), included unmodified in
`data/movies/movies.cypher`.
