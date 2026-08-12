"""
Prompt Builder Module for CARIVIX AI RAG Pipeline
===================================================

Constructs context-aware prompts for the LLM by combining retrieved
document chunks with the user's query.

Prompt Template:

    You are a helpful AI assistant for the CARIVIX AI project.
    Answer the question based ONLY on the provided context.
    If the answer is not available in the context, reply:
    "Information not found."

    Context:
    ---
    [Retrieved Chunk 1]
    ---
    [Retrieved Chunk 2]
    ---
    ...

    Question:
    [User Query]

    Answer:

Usage:
    builder = PromptBuilder()
    prompt = builder.build_prompt(query, retrieved_chunks)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CARIVIX_AI")


class PromptBuilder:
    """
    Builds structured prompts for RAG-based question answering.

    Combines retrieved document chunks as context with the user's
    query to create a prompt that instructs the LLM to answer only
    based on the provided context.

    Supports customizable templates, system messages, and formatting.
    """

    # Default system prompt template
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful AI assistant for the CARIVIX AI project. "
        "Answer the question based ONLY on the provided context. "
        "If the answer is not available in the context, reply: "
        '"Information not found."'
    )

    # Default prompt template with placeholders
    DEFAULT_PROMPT_TEMPLATE = """{system_prompt}

Context:
---
{context}

Question:
{question}

Answer:"""

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
    ) -> str:
        """
        Build a complete prompt from a query and retrieved chunks.

        Args:
            query: The user's question.
            retrieved_chunks: List of result dictionaries from the Retriever.
            max_context_length: Maximum characters for context.
                If None, no limit is applied.

        Returns:
            Formatted prompt string ready for LLM inference.
        """
        # Format the context from retrieved chunks
        context = self._format_context(
            retrieved_chunks, max_context_length
        )

        # Build the prompt
        prompt = self.template.format(
            system_prompt=self.system_prompt,
            context=context,
            question=query.strip(),
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
        context = self._format_context(
            retrieved_chunks, max_context_length
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n---\n{context}\n---\n\n"
                    f"Question: {query.strip()}\n\n"
                    "Answer based only on the provided context."
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
        Format retrieved chunks into a structured context string.

        Args:
            retrieved_chunks: List of result dictionaries.
            max_length: Maximum characters for context.

        Returns:
            Formatted context string.
        """
        if not retrieved_chunks:
            return "No relevant context available."

        parts = []
        total_length = 0

        for chunk in retrieved_chunks:
            text = chunk.get("text", chunk.get("document", ""))
            if hasattr(text, "page_content"):
                text = text.page_content

            # Build chunk header with metadata
            header = f"Source: {chunk.get('source', 'unknown')}"
            if chunk.get("page"):
                header += f" (Page {chunk['page']})"
            if chunk.get("chunk_id"):
                header += f" [Chunk {chunk['chunk_id']}]"

            # Optionally include score
            if self.include_metadata:
                header += f" | Relevance: {chunk.get('score', 0):.4f}"

            chunk_text = f"{header}\n{text}"

            # Check max length
            if max_length and total_length + len(chunk_text) > max_length:
                remaining = max_length - total_length
                if remaining > 50:
                    parts.append(chunk_text[:remaining] + "...")
                break

            parts.append(chunk_text)
            total_length += len(chunk_text)

        context = "\n\n---\n\n".join(parts)
        return context

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
