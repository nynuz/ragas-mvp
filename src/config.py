"""
Single source of truth for all project constants.
Adjust model names, Qdrant params, and paths here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "amnesty_qa"
RESULTS_DIR = DATA_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Output files
RAG_RESPONSES_PATH = RESULTS_DIR / "rag_responses.json"
RAGAS_SCORES_PATH = RESULTS_DIR / "ragas_scores.csv"
ABLATION_TOP_K_PATH = RESULTS_DIR / "ablation_top_k.csv"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "vibrantlabsai/amnesty_qa"
DATASET_CONFIG = "english_v2"
DATASET_SPLIT = "eval"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "amnesty_qa"

# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

GENERATOR_MODEL = "gpt-4.1-mini"
TOP_K_DEFAULT = 3
TOP_K_ABLATION = [3, 5, 10]

# ---------------------------------------------------------------------------
# RAGAS evaluator
# ---------------------------------------------------------------------------

EVALUATOR_MODEL = "gpt-5.1"
RAGAS_MAX_WORKERS = 4                   # throttle concurrent evaluator calls
