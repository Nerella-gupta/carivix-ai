"""
Prompt Builder Module for CARIVIX AI RAG Pipeline
===================================================

Constructs context-aware prompts for the LLM by combining retrieved
document chunks with the user's query.

Prompt Template:

    SYSTEM INSTRUCTIONS

    You are a grounded AI assistant for the CARIVIX AI project.
    Answer using the provided context only. Do not use unsupported information.
    Avoid hallucination. State when the context is insufficient. Keep answers relevant
    and use the retrieved context as the primary source. If the answer is not available in the context, reply: "Information not found."

    CONTEXT:
    ---
    [Retrieved Chunk 1]
    ---
    [Retrieved Chunk 2]
    ---
    ...

    USER QUESTION:
    [User Query]

    ANSWER:

Usage:
    builder = PromptBuilder()
    prompt = builder.build_prompt(query, retrieved_chunks)
"""

import logging
from typing import Any, Dict, List, Optional

from rag.context_builder import ContextBuilder

logger = logging.getLogger("CARIVIX_AI")

class PromptBuilder:
    """
    Builds structured prompts for RAG-based question answering.

    Combines retrieved document chunks as context with the user's
    query to create a prompt that instructs the LLM to answer only
    based on the provided context.

    Supports customizable templates, system messages, and formatting.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a grounded AI assistant for the CARIVIX AI project. "
        "Answer using the provided context only. Do not use unsupported information. "
        "Avoid hallucination. State when the context is insufficient. Keep answers relevant "
        "and use the retrieved context as the primary source. If the answer is not available in the context, reply: \"Information not found.\""
    )

    DEFAULT_PROMPT_TEMPLATE = """SYSTEM INSTRUCTIONS

{system_prompt}

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        include_metadata: bool = False,
    ) -> None:
        """
        Initialize the PromptBuilder.

        Args:
            system_prompt: Custom system prompt. If None, uses DEFAULT_SYSTEM_PROMPT.
            template: Custom prompt template with placeholders
                {system_prompt}, {context}, {question}.
                If None, uses DEFAULT_PROMPT_TEMPLATE.
            include_metadata: If True, includes source/chunk metadata in context.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.template = template or self.DEFAULT_PROMPT_TEMPLATE
        self.include_metadata = include_metadata
        self.context_builder = ContextBuilder(include_metadata=self.include_metadata)

        logger.info(
            "PromptBuilder initialized. "
            "Include metadata: %s",
            include_metadata,
        )

    # ------------------------------------------------------------------
    # Prompt Construction
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_length: Optional[int] = None,
        interpretations: Optional[List[str]] = None,
    ) -> str:
        """
        Build a complete prompt from a query and retrieved chunks.

        Args:
            query: The user's question.
            retrieved_chunks: List of result dictionaries from the Retriever.
            max_context_length: Maximum characters for context.
                If None, no limit is applied.
            interpretations: Optional list of distinct interpretations detected in context.

        Returns:
            Formatted prompt string ready for LLM inference.
        """
        context = self._format_context(retrieved_chunks, max_context_length)
        question_text = query.strip()
        if interpretations:
            interp_list = "\n".join(f"- {item}" for item in interpretations)
            question_text = (
                f"{question_text}\n\n"
                f"IMPORTANT: The context provides information covering multiple distinct interpretations/facets:\n"
                f"{interp_list}\n"
                f"Your answer MUST explicitly address each of these distinct aspects based on the context."
            )

        prompt = self.template.format(
            system_prompt=self.system_prompt,
            context=context,
            question=question_text,
        )

        logger.debug(
            "Prompt built. Context length: %d chars, "
            "Total prompt length: %d chars",
            len(context),
            len(prompt),
        )

        return prompt

    def build_messages(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_length: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a chat-style message list for models with a chat API.

        Args:
            query: The user's question.
            retrieved_chunks: List of result dictionaries from the Retriever.
            max_context_length: Maximum characters for context.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        context = self._format_context(retrieved_chunks, max_context_length)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"USER QUESTION:\n{query.strip()}\n\n"
                    "ANSWER:"
                ),
            },
        ]

        logger.debug(
            "Chat messages built. %d messages, context: %d chars",
            len(messages),
            len(context),
        )

        return messages

    # ------------------------------------------------------------------
    # Context Formatting
    # ------------------------------------------------------------------

    def _format_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        max_length: Optional[int] = None,
    ) -> str:
        """
        Format retrieved chunks into a structured context string using ContextBuilder.

        Args:
            retrieved_chunks: List of result dictionaries.
            max_length: Maximum characters for context.

        Returns:
            Formatted context string.
        """
        return self.context_builder.build(
            retrieved_chunks=retrieved_chunks,
            max_context_chars=max_length,
            include_metadata=self.include_metadata,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_prompt_stats(self, prompt: str) -> Dict[str, int]:
        """
        Get statistics about a built prompt.

        Args:
            prompt: The prompt string to analyze.

        Returns:
            Dictionary with character count, word count, and line count.
        """
        return {
            "characters": len(prompt),
            "words": len(prompt.split()),
            "lines": prompt.count("\n") + 1,
        }

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Update the system prompt.

        Args:
            system_prompt: New system prompt text.
        """
        self.system_prompt = system_prompt
        logger.info("System prompt updated.")

    def set_template(self, template: str) -> None:
        """
        Update the prompt template.

        Args:
            template: New template with {system_prompt}, {context}, {question}.
        """
        self.template = template
        logger.info("Prompt template updated.")

