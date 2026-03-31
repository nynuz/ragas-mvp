# RAG Evaluation with RAGAS

A end-to-end project that builds a RAG pipeline from scratch and evaluates it systematically using the RAGAS framework. The goal is not just to build a working RAG system, but to understand what the numbers actually mean — where the pipeline is strong, where it fails, and how retrieval parameters affect quality.

## Why evaluation matters

Building a RAG system is relatively straightforward. Knowing whether it is actually working is not. A pipeline can produce fluent, confident-sounding answers while consistently missing the point of the question, hallucinating details, or retrieving irrelevant context. Without quantitative metrics, these failure modes are invisible.

RAGAS provides a structured way to measure six distinct aspects of RAG quality: how precisely the retriever selects relevant chunks, how well it covers the necessary information, how faithful the generator is to the provided context, how relevant the answer is to the question, and how factually correct and semantically similar the final answer is to a ground truth reference. Each metric tells a different part of the story.

## What this project does

The pipeline indexes the `vibrantlabsai/amnesty_qa` dataset into a local Qdrant instance using hybrid search (dense semantic vectors + SPLADE sparse vectors with RRF fusion), then generates answers with GPT-4.1-mini and evaluates them with RAGAS. The main experiment is an ablation study across three retrieval configurations (top_k = 3, 5, 10) to measure the tradeoff between context precision and context recall.

## Stack

- **Vector database**: Qdrant (local Docker), HybridSearch with RRF fusion
- **Dense embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (dim=384)
- **Sparse embeddings**: `naver/splade-cocondenser-ensembledistil` via sentence-transformers
- **Generator LLM**: GPT-4.1-mini (OpenAI)
- **Evaluator LLM**: configurable via `src/config.py`
- **Evaluation framework**: RAGAS [https://www.ragas.io/](https://www.ragas.io/)
- **RAG chain**: LangChain (prompt template + LLM + output parser)
- **Package manager**: uv

## Project structure

```
ragas-mvp/
├── notebooks/
│   └── ragas_pipeline.ipynb    # full pipeline: indexing, generation, evaluation, plots
├── src/
│   ├── config.py               # all constants (models, paths, Qdrant params)
│   ├── rag_pipeline.py         # index_corpus(), retrieve(), build_rag_chain()
│   └── evaluation.py           # generate_responses(), run_ragas_evaluation()
├── data/
│   ├── raw/amnesty_qa/         # HuggingFace dataset cache
│   └── results/                # rag_responses.json, ragas_scores.csv, ablation_top_k.csv
├── reports/
│   ├── figures/                # radar chart, line chart, bar chart
│   └── evaluation_report.md   # LLM-generated report
├── docker-compose.yml
└── pyproject.toml
```

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/nynuz/ragas-mvp.git
cd ragas-mvp
uv sync
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Start Qdrant**

```bash
docker compose up -d
```

Qdrant will be available at `http://localhost:6333`. The dashboard is at `http://localhost:6333/dashboard`.

**4. Run the notebook**

```bash
uv run jupyter notebook
```

Open `notebooks/ragas_pipeline.ipynb` and run cells in order.

## Configuration

All parameters are centralized in `src/config.py`. The most relevant ones:

```python
GENERATOR_MODEL = "gpt-4.1-mini"    # LLM used to generate answers
EVALUATOR_MODEL = "gpt-5.1"         # LLM used as RAGAS judge — change as needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_DEFAULT   = 3
TOP_K_ABLATION  = [3, 5, 10]
```

## Metrics

One of the main reasons to use RAGAS is that it separates retriever quality from generator quality. Most evaluation approaches measure the final answer against a reference, which conflates two very different failure modes: the retriever not finding the right information, and the generator not using it correctly. RAGAS breaks this down into distinct signals.

The framework uses an LLM-as-judge approach for most metrics: a separate evaluator model reads the question, the retrieved contexts, the generated response, and the reference answer, then scores each aspect independently. This makes evaluation scalable without requiring human annotators for every sample.

The six metrics used in this project and what they actually measure:

**Context Precision** measures what fraction of the retrieved chunks were actually useful for answering the question. A low score means the retriever is bringing in noise — chunks that are topically related but do not contribute to the answer. This wastes the LLM's context window and can introduce irrelevant information that steers the response in the wrong direction.

**Context Recall** measures how much of the information needed to answer the question was present in the retrieved chunks. A low score means the retriever missed relevant passages — the LLM cannot answer correctly because the evidence was never provided. Precision and recall are in tension: retrieving more chunks improves recall but risks reducing precision, which is exactly what the ablation study in this project measures.

**Faithfulness** measures whether the generated answer is grounded in the retrieved contexts. Specifically, it checks whether each claim made in the response can be traced back to something stated in the context. A high faithfulness score means the model is not hallucinating — it is answering from what it was given. A low score indicates the model is filling gaps with its parametric knowledge, which may or may not be accurate.

**Answer Relevancy** measures whether the response actually addresses the question asked. A model can be perfectly faithful to the context while still producing an answer that is tangential or incomplete. This metric catches responses that are factually grounded but miss the point.

**Factual Correctness** compares the claims in the generated response against the reference answer using an F1-style score. Unlike semantic similarity, it operates at the level of individual facts: how many correct facts are present, how many are missing, and how many are wrong. It is the strictest metric because it requires both the retriever to have found the right information and the generator to have reproduced it accurately.

**Semantic Similarity** measures the embedding-space distance between the generated response and the reference answer. It captures whether the two texts convey the same meaning even if they use different words. It is less strict than factual correctness and more forgiving of paraphrasing, but also less sensitive to factual errors that are semantically adjacent to the truth.

Together these metrics form a diagnostic framework. If faithfulness is low, the problem is the generator or the prompt. If context recall is low, the problem is the retriever configuration. If factual correctness is low while faithfulness is high, the retriever is finding the wrong chunks — the generator is doing its job correctly but working with insufficient evidence.

## Results

The ablation study across top_k values produced the following aggregate scores:

| k  | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|----|-------------------|----------------|--------------|------------------|
| 3  | 0.654             | 0.750          | 0.935        | 0.977            |
| 5  | 0.703             | 0.900          | 0.957        | 0.973            |
| 10 | 0.676             | 1.000          | 0.970        | 0.976            |

The generator performs well across all configurations — faithfulness stays above 0.93 and answer relevancy above 0.97 regardless of k. The retriever is the bottleneck: context recall goes from 0.75 at k=3 to perfect coverage at k=10, but precision peaks at k=5 before declining as irrelevant chunks are introduced.

k=5 is the optimal configuration for this dataset, balancing precision (0.703) and recall (0.900) with high faithfulness (0.957). The counterintuitive result is that faithfulness increases with k rather than decreasing — more context gives the model better grounding and reduces the need to fill gaps.

## Dataset

`vibrantlabsai/amnesty_qa` (config: `english_v2`) — 20 question-answer pairs based on Amnesty International reports. Each sample includes a question, a ground truth answer, a generated response, and three pre-retrieved context passages. The dataset is designed for RAG evaluation and provides a realistic mix of factual questions requiring precise retrieval.
