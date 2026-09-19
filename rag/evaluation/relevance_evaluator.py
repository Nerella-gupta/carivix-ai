"""Retrieval relevance evaluation for RAG queries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("CARIVIX_AI")


class RelevanceEvaluator:
    """Evaluate whether retrieved chunks are relevant to the user's question.

    Uses the pipeline's own FAISS similarity score as the primary signal,
    since that score already reflects semantic similarity (via embeddings),
    not just literal keyword overlap. A pure keyword-overlap metric was
    found to significantly understate relevance for correctly-retrieved
    chunks when the query and chunk share few literal words but are
    semantically related (e.g. a meta-question like "which source
    supports X" versus a chunk that directly states X).

    Keyword overlap is retained as a secondary diagnostic value only.

    Relevance threshold (0.4) is calibrated from real observed FAISS
    scores during Day 1-3 testing: on-topic queries against this test
    corpus scored 0.44-0.72, while a deliberately out-of-domain query
    scored 0.37 or below across all retrieved chunks.
    """

    RELEVANCE_THRESHOLD = 0.4

    def __init__(self) -> None:
        self.logger = logger

    def evaluate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "query": query,
                "retrieved_documents": [],
                "relevance_score": 0.0,
                "lexical_overlap_score": 0.0,
                "relevance_status": "empty_query",
            }

        if not retrieved_chunks:
            return {
                "query": query,
                "retrieved_documents": [],
                "relevance_score": 0.0,
                "lexical_overlap_score": 0.0,
                "relevance_status": "no_context",
            }

        query_terms = set(tok for tok in query.lower().split() if tok.isalpha())

        # Primary signal: rank-weighted FAISS similarity score, already
        # computed by the pipeline during retrieval.
        semantic_scores: List[float] = []
        lexical_scores: List[float] = []
        weights: List[float] = []

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("text") or chunk.get("document")
            if hasattr(text, "page_content"):
                text = text.page_content
            text = str(text or "").lower()

            semantic_score = float(chunk.get("score", 0.0))

            if not query_terms:
                lexical_score = 0.0
            else:
                overlap = sum(1 for term in query_terms if term in text)
                lexical_score = overlap / max(len(query_terms), 1)

            weight = 1.0 / rank
            semantic_scores.append(semantic_score * weight)
            lexical_scores.append(lexical_score * weight)
            weights.append(weight)

        total_weight = sum(weights) if weights else 1.0
        overall_semantic = sum(semantic_scores) / total_weight
        overall_lexical = sum(lexical_scores) / total_weight

        status = "relevant" if overall_semantic >= self.RELEVANCE_THRESHOLD else "not_relevant"

        return {
            "query": query,
            "retrieved_documents": [chunk.get("source") or "unknown" for chunk in retrieved_chunks],
            "relevance_score": round(float(overall_semantic), 4),
            "lexical_overlap_score": round(float(overall_lexical), 4),
            "relevance_status": status,
        }