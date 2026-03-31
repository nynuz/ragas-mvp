# Import e Variabili
import uuid
import numpy as np
from tqdm import tqdm
from src.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K_DEFAULT, GENERATOR_MODEL

from sentence_transformers import SentenceTransformer, SparseEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
    NamedVector,
    NamedSparseVector,
    Prefetch,
    FusionQuery,
    Fusion
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# Client e modelli
def _get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def _get_dense_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)

def _get_sparse_model() -> SentenceTransformer:
    return SentenceTransformer("naver/splade-cocondenser-ensembledistil")


# Pipeline RAG
def index_corpus(texts: list[str]) -> None:
    client = _get_client()

    # Controlliamo che la collection esiste
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        count = client.count(QDRANT_COLLECTION).count
        if count == len(texts):
            print(f"Collection '{QDRANT_COLLECTION}' già esistente ({count} vettori). Skip.")
            return
        print(f"Collection esistente con {count} vettori (attesi {len(texts)}). Re-indicizzazione...")
        client.delete_collection(QDRANT_COLLECTION)

    # Creiamo la collection con dense + sparse
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams()
        }
    )

    # Calcoliamo gli embeddings localmente
    print("Calcolo dense embeddings...")
    dense_model = _get_dense_model()
    dense_vectors = dense_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    print("Calcolo sparse embeddings...")
    sparse_model = _get_sparse_model()
    sparse_vectors = sparse_model.encode(texts, convert_to_tensor=False)

    # Ingestion nel vector db con Upsert
    points = []
    for i, (text, dvec, svec) in enumerate(tqdm(zip(texts, dense_vectors, sparse_vectors), desc="Upsert")):
        nonzero_mask = svec > 0
        indices = np.where(nonzero_mask)[0].tolist()
        values = svec[nonzero_mask].tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dvec.tolist(),
                    SPARSE_VECTOR_NAME: SparseVector(indices=indices, values=values)
                },
                payload={
                    "chunk_id": i,
                    "text": text
                }
            )
        )
    
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    print(f"Indicizzati {len(points)} chunk in '{QDRANT_COLLECTION}'.")

def retrieve(query: str, top_k: int = TOP_K_DEFAULT) -> list[str]:
    client = _get_client()
    dense_model = _get_dense_model()
    sparse_model = _get_sparse_model()

    # Calcoliamo query embeddings
    dvec = dense_model.encode(query, normalize_embeddings=True)
    svec = sparse_model.encode(query, convert_to_tensor=False)
    nonzero_mask = svec > 0
    indices = np.where(nonzero_mask)[0].tolist()
    values = svec[nonzero_mask].tolist()

    # Hybrid search con RRF
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=dvec.tolist(),
                limit=top_k*2,
                using=DENSE_VECTOR_NAME
            ),
            Prefetch(
                query=SparseVector(indices=indices, values=values),
                limit=top_k*2,
                using=SPARSE_VECTOR_NAME
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True
    )

    return [point.payload["text"] for point in results.points]

def build_rag_chain(top_k: int = TOP_K_DEFAULT):
    llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the information provided in the context below."
            "If the context does not contain enough information to answer the question, say so explicitly. Do not use any prior knowledge."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    llm_chain = prompt | llm | StrOutputParser()

    def run_chain(question: str) -> dict:
        context = retrieve(question, top_k)
        context_str = "\n".join(context)
        response = llm_chain.invoke({"context": context_str, "question": question})
        return {
            "response": response,
            "retrieved_context": context
        }

    return run_chain