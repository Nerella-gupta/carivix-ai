"""Groundedness / factuality evaluation for generated answers."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("CARIVIX_AI")


class FactualityEvaluator:
    """Identify supported and unsupported claims based on retrieved context."""

    _STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",
        # Common framing and conversational markers
        "also", "additionally", "specifically", "according", "accordingly", "including",
    }

    _REFUSAL_PHRASES = (
        "information not found",
        "not found in the provided context",
        "no answer available",
        "cannot be answered from the provided context",
        "does not contain",
        "i don't have enough information",
        "insufficient information",
    )

    def __init__(self, min_content_word_ratio: float = 0.75) -> None:
        self.logger = logger
        self.min_content_word_ratio = min_content_word_ratio

    def evaluate(self, answer: str, context: List[str], query: str) -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "query": query,
                "supported_claims": [],
                "unsupported_claims": ["No answer generated."],
                "missing_context": True,
                "potential_hallucinations": ["No answer available for factual evaluation."],
                "declined_to_answer": False,
            }

        answer_lower = answer.strip().lower()
        if any(phrase in answer_lower for phrase in self._REFUSAL_PHRASES):
            return {
                "query": query,
                "supported_claims": [],
                "unsupported_claims": [],
                "missing_context": not bool(context and "".join(context).strip()),
                "potential_hallucinations": [],
                "declined_to_answer": True,
            }

        joined_context = "\n".join(context).lower()
        supported: List[str] = []
        unsupported: List[str] = []

        # Split into sentences or lines, without splitting on decimal numbers (e.g., 0.5837)
        raw_parts = re.split(r"(?<!\d)\.(?!\d)|[\r\n]+", answer)
        sentences = [part.strip() for part in raw_parts if part.strip()]

        for sentence in sentences:
            tokens = [
                re.sub(r"^[^\w]+|[^\w]+$", "", token.lower())
                for token in sentence.split()
            ]
            content_words = [
                t for t in tokens if t and t not in self._STOPWORDS and len(t) > 1
            ]

            if not content_words:
                continue

            supported_words = [w for w in content_words if w in joined_context]
            ratio = len(supported_words) / len(content_words)

            if ratio >= self.min_content_word_ratio:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        missing_context = bool(not context or not joined_context.strip())
        returns = {
            "query": query,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "missing_context": missing_context,
            "potential_hallucinations": unsupported if unsupported else [],
            "declined_to_answer": False,
        }
        return returns
