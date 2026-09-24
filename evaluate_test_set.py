"""
CARIVIX AI - Test Set Evaluation Script
=========================================
Runs the verified TestSetGenerator cases through the real RAG pipeline
(actual retrieval + actual LLM generation), then scores each result with
RelevanceEvaluator and FactualityEvaluator. Wires together three
components that exist in rag/evaluation but are not currently used by
evaluate_rag.py.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger
from rag.pipeline import RAGPipeline
from rag.evaluation import TestSetGenerator, RelevanceEvaluator, FactualityEvaluator

logger = logging.getLogger("CARIVIX_AI")


def main():
    setup_logger()

    pipeline = RAGPipeline(
        documents_dir="data/documents",
        vector_store_dir="data/vector_store",
    )
    pipeline.load_index()

    relevance_eval = RelevanceEvaluator()
    factuality_eval = FactualityEvaluator()

    cases = TestSetGenerator.default_cases()

    print("=" * 70)
    print("  TEST SET EVALUATION")
    print(f"  Cases: {len(cases)}")
    print("=" * 70)

    for i, case in enumerate(cases, start=1):
        print(f"\n{'-' * 70}")
        print(f"Case {i}/{len(cases)} [{case.category}]: {case.query}")
        print(f"{'-' * 70}")

        result = pipeline.query(case.query, k=5, max_tokens=512, temperature=0.0)

        retrieved_chunks = result.get("retrieved_chunks", [])
        answer = result.get("response", "")

        rel_result = relevance_eval.evaluate(case.query, retrieved_chunks)

        context_texts = []
        for chunk in retrieved_chunks:
            text = chunk.get("text") or chunk.get("document")
            if hasattr(text, "page_content"):
                text = text.page_content
            context_texts.append(str(text or ""))

        retrieved_sources = [c.get("source") for c in retrieved_chunks if c.get("source")]
        fact_result = factuality_eval.evaluate(
            answer, context_texts, case.query, retrieved_sources=retrieved_sources
        )

        print(f"Expected category: {case.category}")
        print(f"Expected answer:   {case.expected_answer}")
        print(f"Actual answer:     {answer}")
        print(f"Relevance score:   {rel_result['relevance_score']} ({rel_result['relevance_status']})")
        print(f"Supported claims:   {len(fact_result['supported_claims'])}")
        print(f"Unsupported claims: {len(fact_result['unsupported_claims'])}")
        if fact_result["unsupported_claims"]:
            print(f"  -> Potential hallucinations: {fact_result['unsupported_claims']}")
        print(f"Missing context:   {fact_result['missing_context']}")
        print(f"Detection time:    {result.get('detection_time', 0.0)}s")

    print(f"\n{'=' * 70}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()