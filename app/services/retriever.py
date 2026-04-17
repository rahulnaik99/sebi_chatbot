import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

"""
Hybrid Retriever Pipeline
=========================
Stage 1 — Dense retrieval      : Weaviate vector search (top-K by cosine sim)
Stage 2 — BM25 retrieval       : Weaviate keyword search (top-K)
Stage 3 — RRF fusion           : Reciprocal Rank Fusion merges both lists
Stage 4 — MMR diversity filter : Maximal Marginal Relevance deduplicates
Stage 5 — Cross-encoder rerank : sentence-transformers cross-encoder final sort
"""

from typing import List, Dict, Tuple
import numpy as np

from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore

from app.services.get_weaviate import get_weaviate_client
from app.services.get_model import get_embeddings
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

# Cross-encoder loaded once at module level
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.info("Cross-encoder loaded.")
    return _cross_encoder


def _get_vectorstore() -> WeaviateVectorStore:
    client = get_weaviate_client()
    embeddings = get_embeddings()
    return WeaviateVectorStore(
        client=client,
        index_name=settings.WEAVIATE_INDEX_NAME,
        text_key="text",
        embedding=embeddings,
    )


# ---------------------------------------------------------------------------
# Stage 1 & 2: Dense + BM25 retrieval
# ---------------------------------------------------------------------------

def _dense_retrieve(query: str, k: int) -> List[Document]:
    """Standard vector similarity search."""
    vs = _get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    log.info(f"Dense retrieval: {len(docs)} docs")
    return docs


def _bm25_retrieve(query: str, k: int) -> List[Document]:
    """Weaviate BM25 keyword search."""
    vs = _get_vectorstore()
    # Weaviate's LangChain wrapper exposes BM25 via the underlying client
    collection = vs._client.collections.get(settings.WEAVIATE_INDEX_NAME)
    response = collection.query.bm25(
        query=query,
        limit=k,
        return_properties=["text", "source"],
    )
    docs = []
    for obj in response.objects:
        docs.append(
            Document(
                page_content=obj.properties.get("text", ""),
                metadata={"source": obj.properties.get("source", "")},
            )
        )
    log.info(f"BM25 retrieval: {len(docs)} docs")
    return docs


# ---------------------------------------------------------------------------
# Stage 3: RRF Fusion
# ---------------------------------------------------------------------------

def _rrf_fusion(
    dense_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_n: int = 10,
) -> List[Document]:
    """
    Reciprocal Rank Fusion.
    score(d) = sum(1 / (k + rank_i(d))) across all ranked lists.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in [dense_docs, bm25_docs]:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content[:200]  # dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = [doc_map[key] for key in sorted_keys[:top_n]]
    log.info(f"RRF fusion: {len(fused)} docs")
    return fused


# ---------------------------------------------------------------------------
# Stage 4: MMR diversity filter
# ---------------------------------------------------------------------------

def _mmr_filter(
    query: str,
    docs: List[Document],
    k: int = 5,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """
    Maximal Marginal Relevance.
    Balances relevance to query vs diversity among selected docs.
    lambda_mult=1.0 → pure relevance, 0.0 → pure diversity.
    """
    if not docs:
        return []

    embeddings_model = get_embeddings()
    query_emb = np.array(embeddings_model.embed_query(query))
    doc_embs = np.array(embeddings_model.embed_documents([d.page_content for d in docs]))

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    selected_indices = []
    candidate_indices = list(range(len(docs)))

    # First pick: highest relevance to query
    relevance_scores = [cosine(query_emb, doc_embs[i]) for i in candidate_indices]
    best = candidate_indices[int(np.argmax(relevance_scores))]
    selected_indices.append(best)
    candidate_indices.remove(best)

    while len(selected_indices) < k and candidate_indices:
        mmr_scores = []
        for idx in candidate_indices:
            rel = cosine(query_emb, doc_embs[idx])
            redundancy = max(cosine(doc_embs[idx], doc_embs[s]) for s in selected_indices)
            mmr = lambda_mult * rel - (1 - lambda_mult) * redundancy
            mmr_scores.append((idx, mmr))
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    result = [docs[i] for i in selected_indices]
    log.info(f"MMR filter: {len(result)} docs")
    return result


# ---------------------------------------------------------------------------
# Stage 5: Cross-encoder reranking
# ---------------------------------------------------------------------------

def _cross_encoder_rerank(
    query: str,
    docs: List[Document],
    top_k: int = 3,
) -> List[Document]:
    """Score each doc with cross-encoder and return top_k."""
    if not docs:
        return []

    ce = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = ce.predict(pairs)

    ranked: List[Tuple[float, Document]] = sorted(
        zip(scores, docs), key=lambda x: x[0], reverse=True
    )
    result = [doc for _, doc in ranked[:top_k]]
    log.info(f"Cross-encoder rerank: {len(result)} final docs")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_retrieve(query: str) -> List[Document]:
    """
    Full hybrid retrieval pipeline:
    Dense → BM25 → RRF → MMR → Cross-encoder
    Returns final ranked docs ready for context injection.
    """
    log.info(f"Hybrid retrieval for query: '{query[:80]}...'")

    # Stage 1 & 2
    dense_docs = _dense_retrieve(query, k=settings.DENSE_TOP_K)
    bm25_docs = _bm25_retrieve(query, k=settings.BM25_TOP_K)

    # Stage 3: RRF
    fused_docs = _rrf_fusion(dense_docs, bm25_docs, top_n=settings.RRF_TOP_K)

    # Stage 4: MMR
    mmr_docs = _mmr_filter(
        query,
        fused_docs,
        k=settings.MMR_TOP_K,
        lambda_mult=settings.MMR_LAMBDA,
    )

    # Stage 5: Cross-encoder
    final_docs = _cross_encoder_rerank(query, mmr_docs, top_k=settings.CROSS_ENCODER_TOP_K)

    return final_docs
