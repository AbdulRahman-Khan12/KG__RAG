"""Central configuration for KG__RAG.

Every connection setting is read from the environment (or a local ``.env``
file) and nothing else, so the same code runs against a local Neo4j Desktop
database, a Neo4j AuraDB Free instance, or no database at all.

Nothing here is required. With no settings the app runs on the bundled
in-memory copy of the Movies graph and writes answers from templates, which is
what makes ``streamlit run app.py`` work on a fresh clone with zero setup.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_NAME = "KG__RAG"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MOVIES_CYPHER = DATA_DIR / "movies" / "movies.cypher"   # Neo4j's original script
MOVIES_JSON = DATA_DIR / "movies" / "movies.json"       # parsed, loader-friendly
GOLD_FILE = DATA_DIR / "eval" / "questions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader - avoids a python-dotenv dependency.

    Existing environment variables win, so a value exported in the shell is
    never silently replaced by a stale file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def setting(name: str, default: str = "") -> str:
    """Read a setting from the environment, then Streamlit secrets."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:  # Streamlit Community Cloud keeps secrets here instead of the env
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


# --- Neo4j -------------------------------------------------------------------
# Neo4j Desktop:  bolt://localhost:7687   user neo4j
# AuraDB Free:    neo4j+s://<id>.databases.neo4j.io
NEO4J_URI = setting("NEO4J_URI")
NEO4J_USER = setting("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = setting("NEO4J_PASSWORD")
NEO4J_DATABASE = setting("NEO4J_DATABASE", "neo4j")

# --- LLM ---------------------------------------------------------------------
# Any OpenAI-compatible chat-completions endpoint works: OpenAI, Groq,
# OpenRouter, Google Gemini's OpenAI endpoint, or a local Ollama server.
LLM_API_KEY = setting("LLM_API_KEY")
LLM_BASE_URL = setting("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = setting("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TIMEOUT_S = 30
LLM_TEMPERATURE = 0.0   # answers must be grounded, not creative

# --- Retrieval ---------------------------------------------------------------
MAX_RECORDS = 25          # rows handed to the generator; keeps prompts small
RECOMMEND_LIMIT = 5
PATH_MAX_HOPS = 6         # shortestPath bound - the "six degrees" of the graph
FUZZY_CUTOFF = 0.86       # difflib ratio for typo-tolerant entity linking
