"""All settings in one place. Most can be overridden with environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")   # loads TAVILY_API_KEY etc. from .env (real env vars win)

# Paths
DATA_DIR = BASE_DIR / "data"            # where the ZIP is extracted
CHROMA_DIR = BASE_DIR / "chroma_db"     # persistent vector store
COLLECTION_NAME = "ipl_commentary"
TEXT_EXTENSIONS = {".csv"}              # which extracted files to ingest

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")   # swap via env var, e.g. gemma3:1b

# nomic-embed-text works best with these task prefixes
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Chunking (tokens are estimated from word count)
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOKENS_PER_WORD = 1.4
EMBED_BATCH_SIZE = 16

# Retrieval
TOP_K = 3
LLM_NUM_CTX = 1024   # context window for the LLM

# Live web search (Tavily) for matches that are not in the dataset
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = os.getenv("TAVILY_URL", "https://api.tavily.com/search")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
TAVILY_SNIPPET_CHARS = 700                              # keep web snippets short for a 1B model
TAVILY_TIME_RANGE = os.getenv("TAVILY_TIME_RANGE", "")  # "", day, week, month or year
TAVILY_PREFER_DOMAINS = ["cricbuzz.com", "espncricinfo.com", "iplt20.com"]
DATASET_LAST_SEASON = int(os.getenv("DATASET_LAST_SEASON", "2017"))
