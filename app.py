"""KG__RAG - Streamlit interface.

Run with:  streamlit run app.py

Every answer is shown with the evidence behind it: which entities were linked,
which Cypher template ran, the exact parameters, and the rows the graph
returned. The answer is the least interesting part of a RAG system to inspect;
the retrieval that produced it is where errors live.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from kg_rag import __version__, llm
from kg_rag.backend import connect
from kg_rag.dataset import MovieGraph
from kg_rag.evaluate import evaluate, load_gold
from kg_rag.config import DATA_DIR
from kg_rag.pipeline import MovieRAG
from kg_rag.templates import TEMPLATES
from kg_rag.text2cypher import SCHEMA

st.set_page_config(page_title="KG__RAG", page_icon="◉", layout="wide")

st.markdown(
    """
    <style>
      .answer { font-size: 1.18rem; line-height: 1.5; margin: 0.2rem 0 0.4rem 0; }
      .chips span {
        display: inline-block; font-family: ui-monospace, Menlo, monospace;
        font-size: 0.74rem; padding: 0.1rem 0.45rem; margin: 0 0.3rem 0.3rem 0;
        border: 1px solid #C9D3DE; border-radius: 3px; color: #3B4B5C; background: #F2F5F8;
      }
      .chips span.warn { border-color: #D97706; color: #92400E; background: #FEF3C7; }
      .q { font-weight: 600; margin-top: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- state ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to the graph ...")
def get_backend():
    return connect()


backend, backend_warning = get_backend()
if "rag" not in st.session_state:
    st.session_state.rag = MovieRAG(backend)
    st.session_state.history = []
rag: MovieRAG = st.session_state.rag
graph: MovieGraph = rag.graph


# --- sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### KG__RAG")
    st.caption("Question answering over the Neo4j Movies knowledge graph")
    st.markdown("**Graph backend**")
    if backend.is_neo4j:
        st.success(f"Neo4j — {backend.detail}")
    else:
        st.info("Offline — bundled Movies graph in memory. Set `NEO4J_URI` in `.env` "
                "to query a real Neo4j database.")
    if backend_warning:
        st.warning(backend_warning)
    st.markdown("**Answer generator**")
    st.write(f"LLM: {llm.describe()}" if rag.use_llm else
             "Template sentences (no LLM key set). Add `LLM_API_KEY` for LLM answers.")
    st.divider()
    st.markdown("**Try asking**")
    examples = ["Who acted in The Matrix?", "Who directed it?", "How is Kevin Bacon connected to Tom Cruise?",
                "Recommend movies like Apollo 13", "What did reviewers think of The Replacements?",
                "Which movies came out in 1999?", "Who acted in a movie they also directed?"]
    for ex in examples:
        if st.button(ex, width="stretch"):
            st.session_state.pending = ex
    if st.button("Clear conversation", type="secondary", width="stretch"):
        rag.reset()
        st.session_state.history = []
    st.caption(f"v{__version__}")


tab_ask, tab_explore, tab_templates, tab_schema, tab_eval = st.tabs(
    ["Ask", "Graph explorer", "Cypher templates", "Schema", "Evaluation"])


# --- ask ---------------------------------------------------------------------------
def chips(result) -> str:
    parts = [f"template: {result.route.template or '—'}",
             f"backend: {result.backend}", f"answer: {result.generator}",
             f"{len(result.rows)} rows", f"{result.ms:.0f} ms"]
    html_parts = [f"<span>{html.escape(p)}</span>" for p in parts]
    if result.context_used:
        html_parts.append(f"<span>context {html.escape(result.context_used)}</span>")
    if result.ungrounded:
        html_parts.append(f"<span class='warn'>ungrounded: {html.escape(', '.join(result.ungrounded))}</span>")
    return "<div class='chips'>" + "".join(html_parts) + "</div>"


def show_result(result, expanded: bool) -> None:
    st.markdown(f"<div class='q'>Q: {html.escape(result.question)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{html.escape(result.answer)}</div>", unsafe_allow_html=True)
    st.markdown(chips(result), unsafe_allow_html=True)
    if result.error:
        st.warning(result.error)
    with st.expander("Evidence: linking, Cypher and rows", expanded=expanded):
        left, right = st.columns([2, 3])
        with left:
            st.markdown("**Entity linking**")
            if result.linked.mentions:
                st.dataframe(pd.DataFrame([{"said": m.surface, "linked to": m.value, "type": m.kind,
                                            "fuzzy": m.fuzzy} for m in result.linked.mentions]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No movie or person mentioned.")
            st.markdown(f"**Routing** — {result.route.reason}")
            if result.params:
                st.code(json.dumps(result.params, ensure_ascii=False), language="json")
        with right:
            st.markdown("**Cypher**")
            st.code(result.cypher or "-- no query was run", language="cypher")
        st.markdown(f"**Rows returned** ({len(result.rows)})")
        if result.rows:
            st.dataframe(pd.DataFrame(result.rows), hide_index=True, width="stretch")


with tab_ask:
    question = st.chat_input("Ask about movies, actors, directors, reviews ...")
    question = question or st.session_state.pop("pending", None)
    if question:
        st.session_state.history.append(rag.ask(question))
    if not st.session_state.history:
        st.markdown("#### Ask a question about the Movies graph")
        st.caption("38 movies, 133 people and 253 relationships. Follow-ups such as "
                   "“who directed it?” reuse the previous question's subject.")
    for i, result in enumerate(reversed(st.session_state.history)):
        show_result(result, expanded=(i == 0))
        st.divider()


# --- graph explorer ----------------------------------------------------------------
def neighbourhood_dot(name: str) -> str:
    def q(s: str) -> str:
        return '"' + s.replace('"', '\\"') + '"'
    colours = {"ACTED_IN": "#2563EB", "DIRECTED": "#B45309", "PRODUCED": "#7C3AED",
               "WROTE": "#047857", "REVIEWED": "#BE123C", "FOLLOWS": "#64748B"}
    lines = ["digraph G {", "rankdir=LR; bgcolor=transparent; ranksep=1.2;",
             'node [fontname="Helvetica", fontsize=10, style=filled];',
             'edge [fontname="Helvetica", fontsize=8];']
    if name in graph.movies:
        lines.append(f'{q(name)} [shape=box, fillcolor="#FDE68A"];')
        for e in graph.by_movie.get(name, []):
            label = e.type + (f"\\n{', '.join(e.props.get('roles', []))}" if e.props.get("roles") else "")
            lines.append(f'{q(e.person)} [shape=ellipse, fillcolor="#DBEAFE"];')
            lines.append(f'{q(e.person)} -> {q(name)} [label={q(label)}, color="{colours[e.type]}"];')
    else:
        lines.append(f'{q(name)} [shape=ellipse, fillcolor="#FDE68A"];')
        for e in graph.by_person.get(name, []):
            target = e.movie or e.target
            shape = "box" if e.movie else "ellipse"
            lines.append(f'{q(target)} [shape={shape}, fillcolor="#E5E7EB"];')
            lines.append(f'{q(name)} -> {q(target)} [label={q(e.type)}, color="{colours[e.type]}"];')
    lines.append("}")
    return "\n".join(lines)


with tab_explore:
    kind = st.radio("Explore a", ["Movie", "Person"], horizontal=True)
    options = sorted(graph.movies) if kind == "Movie" else sorted(graph.people)
    default = options.index("The Matrix") if kind == "Movie" else options.index("Tom Hanks")
    focus = st.selectbox(kind, options, index=default)
    info = graph.movies.get(focus) or graph.people.get(focus)
    meta = {k: v for k, v in info.items() if k != "label"}
    st.caption(" · ".join(f"{k}: {v}" for k, v in meta.items()))
    st.graphviz_chart(neighbourhood_dot(focus))


# --- templates ---------------------------------------------------------------------
with tab_templates:
    st.markdown("#### Cypher query templates")
    st.caption("Each question is routed to one of these parameterised queries. User text only ever "
               "enters a query as a parameter, so a question cannot inject Cypher.")
    st.dataframe(pd.DataFrame([{"template": t.id, "example question": t.example,
                                "parameters": ", ".join("$" + s for s in t.slots) or "—"}
                               for t in TEMPLATES.values()]), hide_index=True, width="stretch")
    pick = st.selectbox("Show the query for", list(TEMPLATES), format_func=lambda k: TEMPLATES[k].title)
    st.code(TEMPLATES[pick].cypher, language="cypher")
    if backend.is_neo4j:
        st.markdown("**Query lab** — run read-only Cypher against Neo4j")
        query = st.text_area("Cypher", "MATCH (p:Person)-[:DIRECTED]->(m:Movie)\nRETURN p.name AS director, "
                             "count(m) AS movies ORDER BY movies DESC", height=110)
        if st.button("Run"):
            try:
                st.dataframe(pd.DataFrame(rag.run_cypher(query)), width="stretch")
            except Exception as exc:
                st.error(str(exc))


# --- schema --------------------------------------------------------------------------
with tab_schema:
    stats = backend.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", stats["movies"])
    c2.metric("People", stats["people"])
    c3.metric("Relationships", stats["relationships"])
    st.graphviz_chart(
        """digraph S { rankdir=LR; bgcolor=transparent;
        node [fontname="Helvetica", style=filled, fontsize=11];
        Person [shape=ellipse, fillcolor="#DBEAFE", label="Person\\nname, born"];
        Movie [shape=box, fillcolor="#FDE68A", label="Movie\\ntitle, released, tagline"];
        Person -> Movie [label="ACTED_IN {roles}"]; Person -> Movie [label="DIRECTED"];
        Person -> Movie [label="PRODUCED"]; Person -> Movie [label="WROTE"];
        Person -> Movie [label="REVIEWED {summary, rating}"]; Person -> Person [label="FOLLOWS"]; }""")
    st.bar_chart(pd.Series(stats["by_type"], name="relationships"), horizontal=True)
    st.code(SCHEMA, language="text")


# --- evaluation --------------------------------------------------------------------
with tab_eval:
    st.markdown("#### Evaluation")
    st.caption("Each question is scored on routing, slot filling, retrieval and answer separately, "
               "so a failure points at the stage that caused it. The stress set was written after the "
               "router was finished and never used to tune it.")
    which = st.radio("Question set", ["Gold (32)", "Stress paraphrases (15)"], horizontal=True)
    path = DATA_DIR / "eval" / ("questions.json" if which.startswith("Gold") else "stress.json")
    if st.button("Run evaluation", type="primary"):
        report = evaluate(MovieRAG(backend, use_llm=rag.use_llm), load_gold(path))
        s = report.summary()
        cols = st.columns(6)
        for col, key in zip(cols, ["routing_accuracy", "slot_accuracy", "retrieval_recall",
                                   "answer_recall", "grounded_rate", "end_to_end"]):
            col.metric(key.replace("_", " "), f"{s[key]:.2f}")
        st.dataframe(pd.DataFrame([{"ok": "✓" if c.ok else "✗", "question": c.question,
                                    "expected": c.expected, "predicted": c.predicted,
                                    "routing": c.template_ok, "retrieval": c.retrieval_ok,
                                    "answer": c.answer} for c in report.cases]),
                     hide_index=True, width="stretch")
