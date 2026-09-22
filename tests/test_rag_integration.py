"""Focused regression tests for retrieval evidence and model context wiring."""

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import ai_integration
from api import create_app
from rag.context_builder import ContextBuilder
from rag.prompt_builder import PromptBuilder


class _FakeModelService:
    def list_models(self):
        return []

    def has_loaded_models(self):
        return False


class _FakeRAGPipeline:
    model_name = "test-model"

    def query(self, question, k=None, max_tokens=512, temperature=0.0, verbose=False):
        chunk = {
            "rank": 1,
            "score": 0.91,
            "distance": 0.1,
            "document": Document(
                page_content="CARIVIX AI uses retrieval augmented generation for grounded answers.",
                metadata={
                    "source": "rag_workflow.md",
                    "document_id": "doc-001",
                    "file_name": "rag_workflow.md",
                    "chunk_id": "doc-001-0",
                    "chunk_index": 0,
                },
            ),
            "source": "rag_workflow.md",
            "document_id": "doc-001",
            "file_name": "rag_workflow.md",
            "chunk_id": "doc-001-0",
            "chunk_index": 0,
        }
        context = ContextBuilder().build([chunk])
        prompt = PromptBuilder(include_metadata=True).build_prompt(question, [chunk])
        return {
            "question": question,
            "response": "The answer is grounded in the retrieved workflow document.",
            "retrieved_chunks": [chunk],
            "context": context,
            "prompt": prompt,
            "retrieval_time": 0.01,
            "generation_time": 0.02,
            "total_time": 0.03,
        }


def test_context_builder_preserves_sources_and_deduplicates_text():
    chunks = [
        {
            "text": "Market analysis content.",
            "source": "market.csv",
            "document_id": "market-1",
            "chunk_id": 1,
        },
        {
            "text": "Market analysis content.",
            "source": "market.csv",
            "document_id": "market-1",
            "chunk_id": 1,
        },
    ]

    context = ContextBuilder().build(chunks)

    assert context.count("Market analysis content.") == 1
    assert "Source: market.csv" in context
    assert "Document ID: market-1" in context


def test_prompt_separates_instructions_context_and_query():
    chunks = [{"text": "Economic trend content.", "source": "economic.csv", "score": 0.8}]

    prompt = PromptBuilder(include_metadata=True).build_prompt(
        "What is the economic trend?", chunks
    )

    assert "SYSTEM INSTRUCTIONS" in prompt
    assert "CONTEXT:" in prompt
    assert "Source: economic.csv" in prompt
    assert "Economic trend content." in prompt
    assert "USER QUESTION:" in prompt
    assert "What is the economic trend?" in prompt


def test_ai_endpoint_returns_retrieval_and_model_context_evidence():
    app = create_app(model_service=_FakeModelService())
    app.state.rag_pipeline = _FakeRAGPipeline()

    response = TestClient(app).post(
        "/api/v1/ai/query",
        json={"query": "What does the RAG workflow explain?", "k": 1},
    )

    assert response.status_code == 200
    body = response.json()
    rag_context = body["rag_context"]
    assert body["selected_route"] == "rag"
    assert len(rag_context["retrieved_chunks"]) == 1
    assert rag_context["retrieved_chunks"][0]["text"]
    assert rag_context["retrieved_chunks"][0]["source"] == "rag_workflow.md"
    assert rag_context["retrieved_chunks"][0]["document_id"] == "doc-001"
    assert rag_context["retrieved_chunks"][0]["score"] == 0.91
    assert "file_path" not in rag_context["retrieved_chunks"][0]["metadata"]
    assert rag_context["context_length"] == len(rag_context["context"])
    assert rag_context["prompt_length"] == len(rag_context["prompt"])
    assert "rag_workflow.md" in rag_context["context"]
    assert "CARIVIX AI uses retrieval augmented generation" in rag_context["prompt"]
    assert "What does the RAG workflow explain?" in rag_context["prompt"]
    assert body["final_response"]


def test_ai_endpoint_rejects_invalid_query_controls():
    app = create_app(model_service=_FakeModelService())
    response = TestClient(app).post(
        "/api/v1/ai/query",
        json={"query": "Explain the report", "k": 0, "temperature": 3},
    )

    assert response.status_code == 422
    assert response.json() == {"success": False, "error": "Invalid request data."}


def test_ai_endpoint_sanitizes_rag_failure(monkeypatch):
    app = create_app(model_service=_FakeModelService())

    def fail_pipeline(_app):
        raise RuntimeError("secret index path")

    monkeypatch.setattr(ai_integration, "_get_rag_pipeline", fail_pipeline)
    response = TestClient(app).post(
        "/api/v1/ai/query", json={"query": "Explain the report"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "RAG query failed."
    assert "secret index path" not in response.text


def test_ai_endpoint_reports_missing_rag_index_as_unavailable(monkeypatch):
    app = create_app(model_service=_FakeModelService())

    def fail_pipeline(_app):
        raise RuntimeError("No index available. Run index_documents() first.")

    monkeypatch.setattr(ai_integration, "_get_rag_pipeline", fail_pipeline)
    response = TestClient(app).post(
        "/api/v1/ai/query", json={"query": "Explain the report"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "RAG service is unavailable."


def test_ai_endpoint_sanitizes_nlp_failure(monkeypatch):
    app = create_app(model_service=_FakeModelService())

    def fail_nlp(_query):
        raise RuntimeError("secret classifier state")

    monkeypatch.setattr(ai_integration, "nlp_analyze", fail_nlp)
    response = TestClient(app).post(
        "/api/v1/ai/query", json={"query": "Explain the report"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "AI query processing failed."
    assert "secret classifier state" not in response.text


def test_ai_endpoint_sanitizes_ml_failure(monkeypatch):
    class FailingModelService(_FakeModelService):
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("secret model failure")

    app = create_app(model_service=FailingModelService())
    monkeypatch.setattr(
        ai_integration,
        "nlp_analyze",
        lambda _query: {
            "processed_query": "predict",
            "intent": "ml_query",
            "confidence": 1.0,
            "entities": {},
            "route": "ml",
        },
    )
    response = TestClient(app).post(
        "/api/v1/ai/query",
        json={
            "query": "predict age 43 income 67976 credit score 694",
            "auto_fill_missing": True,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "ML prediction failed."
    assert "secret model failure" not in response.text
