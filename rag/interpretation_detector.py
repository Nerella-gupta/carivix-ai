"""
Interpretation Detector Module for CARIVIX AI RAG Pipeline
============================================================

Analyzes user questions and retrieved context to detect whether
the context contains multiple substantively distinct interpretations,
methodologies, or categories relevant to answering the query.

Used in two-pass generation to ensure multi-faceted questions are
comprehensively answered across all supported dimensions.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CARIVIX_AI")


class InterpretationDetector:
    """Detect whether retrieved context contains multiple distinct interpretations."""

    DETECTOR_SYSTEM_PROMPT = (
        "You are an analytical assistant for the CARIVIX AI platform.\n"
        "Analyze the user question and the provided context to determine if the question has "
        "multiple genuinely distinct, competing interpretations or fundamentally different methodologies answering it.\n\n"
        "STRICT CRITERIA:\n"
        "- Interpretations ONLY count if answering the question ONE way would be materially incomplete "
        "or misleading without the other — i.e., they must be competing, alternative, or mutually exclusive answers "
        "to the SAME question, NOT just multiple distinct facts, features, or components that are all part of one coherent answer.\n"
        "- Complementary definitions, descriptions, or sentences explaining what an entity or platform is (e.g., its overall definition, capabilities, components, and what it integrates) are NOT competing interpretations. They are parts of ONE unified answer. has_multiple_interpretations MUST be false.\n"
        "- If the context does not answer the question (insufficient context), set has_multiple_interpretations to false and interpretations to [].\n"
        "- If the question has a single coherent answer (even if that answer consists of multiple features, list items, or capabilities), "
        "set has_multiple_interpretations to false and interpretations to [].\n\n"
        "EXAMPLES:\n\n"
        "Positive Example (Competing Interpretations):\n"
        'Question: "What models are used for forecasting?"\n'
        "Context: Discusses classical time-series statistical models (ARIMA, Prophet) and machine learning models (Random Forest, XGBoost).\n"
        'Reasoning: Time-series models and machine learning algorithms are two distinct, non-overlapping methodologies answering "what models are used"; answering with only one would be materially incomplete.\n'
        'Output: {"has_multiple_interpretations": true, "interpretations": ["Time-series forecasting models (e.g., ARIMA, Prophet)", "Machine learning models (e.g., Random Forest, XGBoost)"]}\n\n'
        "Negative Example 1 (Single entity definition / description):\n"
        'Question: "What is CARIVIX AI?"\n'
        "Context: States CARIVIX AI is a comprehensive AI platform for economic analysis, and that it integrates machine learning models, data processing pipelines, and advanced analytics.\n"
        "Reasoning: Multiple descriptive sentences, attributes, or capabilities of a platform are all part of one unified definition. They are not competing, alternative, or mutually exclusive answers.\n"
        'Output: {"has_multiple_interpretations": false, "interpretations": []}\n\n'
        "Negative Example 2 (Workflow with incidental details):\n"
        'Question: "How does CARIVIX handle model experiment tracking?"\n'
        "Context: Mentions tracking experiments with MLflow, alongside notes on supported algorithms.\n"
        "Reasoning: MLflow is the single mechanism for experiment tracking; mentioning supported algorithms does not constitute a competing interpretation of experiment tracking.\n"
        'Output: {"has_multiple_interpretations": false, "interpretations": []}\n\n'
        "Negative Example 3 (Unanswered or insufficient context):\n"
        'Question: "What is the projected revenue growth next quarter?"\n'
        "Context: Discusses macroeconomic indicators (GDP growth rates) and technical analysis, but does not provide projected revenue growth.\n"
        "Reasoning: The context does not answer the question; incidental background concepts are not valid interpretations.\n"
        'Output: {"has_multiple_interpretations": false, "interpretations": []}\n\n'
        "Respond ONLY with a JSON object in this exact format:\n"
        "{\n"
        '  "has_multiple_interpretations": true,\n'
        '  "interpretations": ["Interpretation 1", "Interpretation 2"]\n'
        "}\n"
        "OR\n"
        "{\n"
        '  "has_multiple_interpretations": false,\n'
        '  "interpretations": []\n'
        "}"
    )

    def __init__(self, generator: Optional[Any] = None) -> None:
        """
        Initialize the InterpretationDetector.

        Args:
            generator: Configured ResponseGenerator instance.
        """
        self.generator = generator

    def detect(self, question: str, context: str) -> Dict[str, Any]:
        """
        Detect whether the context contains multiple distinct interpretations for the question.

        Args:
            question: The user's query.
            context: Formatted context string.

        Returns:
            Dict containing:
                - has_multiple_interpretations (bool)
                - interpretations (List[str])
                - raw_response (str)
        """
        if not question or not context or context.strip() == "No relevant context available.":
            return {
                "has_multiple_interpretations": False,
                "interpretations": [],
                "raw_response": "",
            }

        if self.generator is None or not self.generator.is_available():
            logger.debug("Generator not available for interpretation detection; skipping.")
            return {
                "has_multiple_interpretations": False,
                "interpretations": [],
                "raw_response": "",
            }

        detector_prompt = (
            f"{self.DETECTOR_SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question.strip()}\n\n"
            f"JSON:"
        )

        try:
            raw_response = self.generator.generate(
                prompt=detector_prompt,
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Interpretation detector generation error: %s", exc)
            return {
                "has_multiple_interpretations": False,
                "interpretations": [],
                "raw_response": "",
            }

        return self._parse_response(raw_response)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse structured JSON response with fallback handling.

        1. Attempt direct json.loads()
        2. Attempt to strip markdown code fences or trailing prose and re-parse
        3. Fall back to single-pass and log warning with raw output
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return {
                "has_multiple_interpretations": False,
                "interpretations": [],
                "raw_response": "",
            }

        # Step 1: Attempt direct json.loads()
        try:
            data = json.loads(cleaned)
            return self._validate_data(data, cleaned)
        except Exception:
            pass

        # Step 2: Attempt to strip common markdown code fences and prose
        stripped = cleaned
        if "```" in stripped:
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)

        # Attempt extraction of { ... }
        match = re.search(r"\{[\s\S]*\}", stripped)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._validate_data(data, cleaned)
            except Exception:
                pass

        # Step 3: Fall back to single-pass and log warning
        logger.warning(
            "Failed to parse interpretation detector response: %r",
            cleaned,
        )
        return {
            "has_multiple_interpretations": False,
            "interpretations": [],
            "raw_response": cleaned,
        }

    def _validate_data(self, data: Any, raw: str) -> Dict[str, Any]:
        """Validate and normalize parsed JSON dictionary."""
        if isinstance(data, dict):
            has_multi = bool(
                data.get("has_multiple_interpretations")
                or data.get("multiple_interpretations")
            )
            raw_interps = data.get("interpretations")
            if isinstance(raw_interps, list):
                interps = [str(item).strip() for item in raw_interps if str(item).strip()]
                if has_multi and len(interps) >= 2:
                    return {
                        "has_multiple_interpretations": True,
                        "interpretations": interps,
                        "raw_response": raw,
                    }

        return {
            "has_multiple_interpretations": False,
            "interpretations": [],
            "raw_response": raw,
        }
