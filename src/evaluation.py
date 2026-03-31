import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from src.config import RAG_RESPONSES_PATH, RESULTS_DIR, RAGAS_SCORES_PATH, EVALUATOR_MODEL, RAGAS_MAX_WORKERS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import funzioni RAGAS
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
    FactualCorrectness,
    SemanticSimilarity
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import RunConfig


def generate_responses(dataset: Dataset, chain) -> list[dict]:
    """Restituisce una lista di dict nel formato atteso da ragas."""
    results = []
    for sample in tqdm(dataset, desc="Generate responses"):
        try:
            output = chain(sample['question'])
            results.append({
                "user_input": sample['question'],
                "response": output['response'],
                "retrieved_contexts": output['retrieved_context'],
                "reference": sample['ground_truth']
            })
        except Exception as e:
            print(f"\nErrore sul sample '{sample['question'][:60]}...': {e}")
            continue

    # Salviamo su disco i risultati
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAG_RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSalvati {len(results)} risultati in {RAG_RESPONSES_PATH}")
    return results

def run_ragas_evaluation(responses: list[dict]) -> pd.DataFrame:
    # Step 1: Creazione EvaluationDataset
    samples = [
        SingleTurnSample(
            user_input=r['user_input'],
            response=r['response'],
            retrieved_contexts=r['retrieved_contexts'],
            reference=r['reference']
        )
        for r in responses
    ]
    dataset = EvaluationDataset(samples=samples)

    # Step 2: Evaluator LLM e embeddings
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings()
    )

    # Step 3: Creazione set di metric
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        FactualCorrectness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings)
    ]

    # Step 4: Esecuzione della valutazione
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS)
    )
    df_result = result.to_pandas()

    # Salviamo i risultati
    RAGAS_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(RAGAS_SCORES_PATH, index=False)

    # Stampa aggregate
    metric_cols = [c for c in df_result.columns if c not in ["user_input", "response", "retrieved_contexts", "reference"]]
    print("\n--- Aggregate Scores ---")
    print(df_result[metric_cols].agg(["mean", "median", "std"]).round(3).to_string())

    return df_result