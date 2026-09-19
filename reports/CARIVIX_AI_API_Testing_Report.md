# CARIVIX AI - Comprehensive API Testing & Postman Validation Report

**Project**: CARIVIX AI  
**Sprint**: Sprint 5  
**Date**: 19 September 2026  
**Execution Environment**: Local Windows / Python 3.14.6 Virtual Environment (`venv`)  
**Prepared By**: Vishwanand Yenuganti (ML & AI Engineer Helper)  
**Collaborating Engineer**: Raghavendra Guptha (ML & AI Engineer)  
**Target API Server**: `http://127.0.0.1:8000` (FastAPI / Uvicorn)  

---

## 1. Executive Summary

This report documents the end-to-end verification and API testing conducted for the CARIVIX AI platform using automated Postman collection execution, pytest regression suite runs, and the RAG evaluation framework.

All testing was conducted against the live, running FastAPI application (`api.py` with mounted `ai_integration.py` routes) and the production-configured RAG pipeline (`rag/pipeline.py` with approved 300/50/3 defaults).

### Test Execution Scorecard

| Test Suite | Scope | Total Tests | Passed | Failed | Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Postman API Collection** | Health, ML Inference, RAG Routing, Errors, OpenAPI | 15 | 15 | 0 | **100.0%** |
| **Automated Unit & Integration** | `test_api.py`, `test_rag_pipeline.py` | 50 | 50 | 0 | **100.0%** |
| **RAG Test-Set Suite** | `evaluate_rag.py --mode test-set` | 5 | 5 | 0 | **100.0%** |
| **Overall Combined Score** | **Full System Verification** | **70** | **70** | **0** | **100.0%** |

---

## 2. Postman Test Artifacts & Setup

To enable repeatable testing and manual exploration, complete Postman v2.1.0 collection and environment files have been generated and validated:

1. **Postman Collection File**:  
   [`postman/CARIVIX_AI.postman_collection.json`](file:///c:/Users/VISHWANAND/CARIVIX_AI_Model_Training/CARIVIX_AI_Model_Training/postman/CARIVIX_AI.postman_collection.json)  
   Contains 15 structured requests across 5 logical folders, with comprehensive JavaScript test scripts asserting status codes, response headers, JSON schemas, model prediction fields, and grounding assertions (`pm.test`, `pm.expect`).

2. **Postman Environment File**:  
   [`postman/CARIVIX_AI.postman_environment.json`](file:///c:/Users/VISHWANAND/CARIVIX_AI_Model_Training/CARIVIX_AI_Model_Training/postman/CARIVIX_AI.postman_environment.json)  
   Configures environment variables including `base_url` (`http://127.0.0.1:8000`), `retrieval_k` (`3`), and `default_model` (`RandomForest`).

3. **Automated Test Runner**:  
   [`run_postman_api_tests.py`](file:///c:/Users/VISHWANAND/CARIVIX_AI_Model_Training/CARIVIX_AI_Model_Training/run_postman_api_tests.py)  
   Executes the entire suite programmatically against the running service, logging detailed latency and response bodies to [`reports/postman_api_test_results.json`](file:///c:/Users/VISHWANAND/CARIVIX_AI_Model_Training/CARIVIX_AI_Model_Training/reports/postman_api_test_results.json).

### How to Import & Run in Postman Desktop / Web
1. Open Postman.
2. Click **Import** in the top-left corner.
3. Drag and drop both `postman/CARIVIX_AI.postman_collection.json` and `postman/CARIVIX_AI.postman_environment.json`.
4. Select the environment **CARIVIX AI Local Environment** in the top-right environment selector.
5. Right-click the **CARIVIX AI API Test Collection** and select **Run collection** to execute all 15 tests in sequence.

---

## 3. Detailed API Test Execution Results

All 15 Postman test cases executed against `http://127.0.0.1:8000`. Below is the complete record of each endpoint, request method, status code, latency, and assertion status.

| Test ID | Category | Method | Endpoint | Expected Status | Actual Status | Latency | Result | Key Assertion |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **TC-01** | Health & System | `GET` | `/health` | 200 | 200 | 8.9 ms | **PASS** | `status == "healthy"`, `models_loaded == 8` |
| **TC-02** | Health & System | `GET` | `/api/v1/models` | 200 | 200 | 3.8 ms | **PASS** | 8 models returned, all status `"loaded"` |
| **TC-03** | Health & System | `GET` | `/api/v1/model/info` | 200 | 200 | 27.5 ms | **PASS** | `LogisticRegression` classification metadata |
| **TC-04** | ML Inference | `POST` | `/api/v1/predict` | 200 | 200 | 239.5 ms | **PASS** | `prediction == 1`, `confidence == 0.6485` |
| **TC-05** | ML Inference | `POST` | `/api/v1/predict` (RF) | 200 | 200 | 206.6 ms | **PASS** | Targeted model `RandomForest` executed |
| **TC-06** | ML Inference | `POST` | `/api/v1/predict/batch` | 200 | 200 | 187.6 ms | **PASS** | 2 records processed, `count == 2` |
| **TC-07** | AI Integration | `POST` | `/api/v1/ai/query` (RAG) | 200 | 200 | 25.20 s | **PASS** | Route `rag`, 3 chunks retrieved, response grounded |
| **TC-08** | AI Integration | `POST` | `/api/v1/ai/query` (Context)| 200 | 200 | 8.05 s | **PASS** | Route `rag`, retrieved ML algorithms context |
| **TC-09** | AI Integration | `POST` | `/api/v1/ai/query` (ML) | 200 | 200 | 4.5 ms | **PASS** | Route `ml`, honest missing fields message |
| **TC-10** | AI Integration | `POST` | `/api/v1/ai/query` (Comb) | 200 | 200 | 7.17 s | **PASS** | Route `combined`, retrieval + synthesis |
| **TC-11** | AI Integration | `POST` | `/api/v1/ai/query` (Refusal)| 200 | 200 | 4.86 s | **PASS** | Declined hallucination: `"Information not found"` |
| **TC-12** | Validation & Errors | `POST` | `/api/v1/ai/query` | 400 | 400 | 3.0 ms | **PASS** | Empty query rejected with HTTP 400 |
| **TC-13** | Validation & Errors | `POST` | `/api/v1/predict` | 422 | 422 | 9.1 ms | **PASS** | Invalid types & missing fields rejected (422) |
| **TC-14** | Docs & Schema | `GET` | `/` | 200 | 200 (307) | 22.7 ms | **PASS** | Root redirects to `/docs` |
| **TC-15** | Docs & Schema | `GET` | `/openapi.json` | 200 | 200 | 69.1 ms | **PASS** | Valid OpenAPI 3.1.0 JSON specification |

---

## 4. In-Depth Endpoint Analysis & Verification

### 4.1. Health and Model Discovery (`/health`, `/api/v1/models`, `/api/v1/model/info`)
- **GET `/health`**: Confirmed `status: "healthy"` and `models_loaded: 8`, `models_failed: 0`. The health probe checks model service state and ensures zero degradation.
- **GET `/api/v1/models`**: Returned all 8 trained models:
  1. `LogisticRegression`
  2. `NaiveBayes`
  3. `DecisionTree`
  4. `GradientBoosting`
  5. `XGBoost`
  6. `SVM`
  7. `KNN`
  8. `RandomForest`
- **GET `/api/v1/model/info`**: Successfully returned default model metadata (`algorithm: "LogisticRegression"`, `task_type: "classification"`, `status: "loaded"`).

### 4.2. Machine Learning Inference Endpoints (`/api/v1/predict`, `/api/v1/predict/batch`)
- **Single Record (`POST /api/v1/predict`)**:  
  Sent customer credit profile:
  ```json
  {
    "age": 43, "income": 67976.67, "credit_score": 694, "loan_amount": 14857.28,
    "years_employed": 19, "education": "Master", "employment_status": "Employed",
    "marital_status": "Married", "housing_type": "Rent", "application_date": "2024-09-21"
  }
  ```
  Response received in 239.5 ms:
  ```json
  {
    "success": true,
    "model": "LogisticRegression",
    "prediction": 1,
    "confidence": 0.6485,
    "probability": 0.6485
  }
  ```
- **Targeted Model Selection (`model: "RandomForest"`)**:  
  Targeted model override verified. The system invoked the RandomForest estimator, returning `prediction: 1` and `confidence: 0.52` in 206.6 ms.
- **Batch Inference (`POST /api/v1/predict/batch`)**:  
  Sent 2 applicant records simultaneously. Vector preprocessing processed both records in 187.6 ms, returning `predictions: [1, 1]` with `count: 2`.

### 4.3. AI Integration & RAG Retrieval (`POST /api/v1/ai/query`)
- **RAG Route (Direct Factual)**:  
  Query: `"What is CARIVIX AI?"`  
  - Route selected: `rag` (confidence 0.6)  
  - Retrieved: 3 chunks from `sample_about_carivix.txt` (top similarity: 0.7521)  
  - Generation: Ollama LLM generated a fully grounded overview ("CARIVIX AI is a comprehensive artificial intelligence platform designed for economic analysis...").
- **RAG Route (Supported ML Algorithms)**:  
  Query: `"Which machine learning algorithms are supported?"`  
  - Retrieved: 3 chunks mentioning Random Forest and XGBoost.  
  - Answer correctly synthesized and cited supported algorithms without extraneous claims.
- **ML Route (Intent Classification & Feature Gap Handling)**:  
  Query: `"Predict the customer churn probability for this account"`  
  - Route selected: `ml`  
  - System honestly responded:  
    `"ML routing selected, but automatic feature extraction did not find all required fields. Please call /api/v1/predict with the full set of model features or enable auto_fill_missing."`  
  - Confirmed the system does not fabricate predictions from unstructured text when features are missing.
- **Combined Route**:  
  Query: `"Explain economic forecasting"`  
  - Intent classification triggered `combined` route.  
  - RAG context retrieved 3 chunks; LLM generated an integrated response.
- **Out-of-Domain Refusal**:  
  Query: `"What is the capital of France?"`  
  - Retrieved chunks had low similarity (<0.37).  
  - Generated response: `"Information not found."`  
  - Confirms the model declines to hallucinate when context is absent.

### 4.4. Input Validation & Error Handling
- **Empty Query (HTTP 400)**:  
  Sending whitespace-only `{"query": "   "}` was rejected with `HTTP 400 Bad Request` and `{"detail": "Empty query"}`.
- **Invalid Schema / Negative Numbers (HTTP 422)**:  
  Sending `{"age": -5, "income": "invalid"}` was rejected with `HTTP 422 Unprocessable Entity` and standard error payload `{"success": false, "error": "Invalid request data."}`.

---

## 5. RAG Test-Set Evaluation Suite (`evaluate_rag.py --mode test-set`)

The test-set evaluation was executed via the newly integrated CLI mode (`evaluate_rag.py --mode test-set`), testing all 5 ground-truth cases across retrieval and generation.

### Detailed Test-Set Results

| Case # | Category | Query | FAISS Relevance | Grounding Status | Declination Status | Outcome |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | `direct-factual` | "What is CARIVIX AI?" | **0.6878** | 2 Supported / 0 Unsupported | Declined: False | **PASS** |
| **2** | `context-based` | "Which machine learning algorithms are supported?" | **0.5202** | 7 Supported / 0 Unsupported | Declined: False | **PASS** |
| **3** | `multi-document` | "How does CARIVIX handle model experiment tracking?" | **0.5344** | 1 Supported / 0 Unsupported | Declined: False | **PASS** |
| **4** | `insufficient-context` | "What is the projected revenue growth next quarter?" | **0.4548** | 0 Supported / 0 Unsupported | **Declined: True** | **PASS** |
| **5** | `source-grounding` | "Which source supports the statement about experiment tracking?" | **0.4828** | 4 Supported / 0 Unsupported | Declined: False | **PASS** |

### Key Observations:
1. **0 False Hallucinations**: In Case 4, the system recognized the absence of context and returned an approved refusal phrase, which was correctly classified with `Declined: True` and zero unsupported claims.
2. **Relevance Robustness**: All 5 cases met or exceeded the calibrated `0.40` semantic relevance threshold.
3. **Report Generation**: The CLI automatically refreshed `evaluation_report.md` with active parameters:
   - Chunk Size: `300`
   - Chunk Overlap: `50`
   - Retrieval Top-k: `3`
   - Processing Device: `CPU` (detected at runtime)

---

## 6. Pytest Regression Test Execution (`tests/`)

The automated regression test suite was executed in the virtual environment:
- **Total Tests Collected**: 50
- **Total Tests Passed**: 50 (100%)
- **Test Modules Exercised**:
  - `tests/test_api.py`: 10 passed (health, model listing, info, single predict, batch predict, invalid input handling, failure modes)
  - `tests/test_rag_pipeline.py`: 40 passed (document loading, text preprocessing, chunking, embeddings, FAISS vector store indexing, similarity retrieval, prompt construction, end-to-end pipeline execution)
- **Execution Time**: 328.70s

---

## 7. Performance & Latency Benchmarks

| Operation | State | Observed Latency | Production Benchmark Target | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Health Check (`/health`)** | Warm | 8.9 ms | < 50 ms | Optimal |
| **Model Info (`/model/info`)** | Warm | 27.5 ms | < 50 ms | Optimal |
| **ML Single Prediction** | Warm | 206.6 – 239.5 ms | < 300 ms | Within Target |
| **ML Batch Prediction (2)** | Warm | 187.6 ms | < 300 ms | High Throughput |
| **FAISS Vector Retrieval** | Warm | 15.1 – 30.9 ms | < 100 ms | Exceptional (sub-35ms) |
| **RAG End-to-End Query** | Cold | 25.20 s | < 30.0 s | SentenceTransformer load |
| **RAG End-to-End Query** | Warm | 4.86 – 8.05 s | < 10.0 s | Dominated by LLM token generation |

---

## 8. Verified Git Commits & Traceability

All improvements and test integrations are committed and pushed to the upstream repository:
- **`33212476`**: Fixed 4 evaluation defects (`RAGPipeline` logging format, `report.py` device detection, `FactualityEvaluator` refusal handling, `RelevanceEvaluator` semantic similarity scoring).
- **`db88fb6c`**: Applied approved 300/50/3 chunking defaults, resolved duplicated CLI defaults in `main_rag.py`, fixed Windows Unicode console encoding issues.
- **`aebbde3c`**: Wired `--mode test-set` into `evaluate_rag.py` CLI and bound system configuration to shared `rag.config` constants.

---

## 9. Conclusion

The CARIVIX AI platform has achieved **100% test pass rates** across all three verification layers:
1. **API Endpoints (Postman Suite)**: 15 / 15 Passed
2. **Automated Unit & Integration (Pytest Suite)**: 50 / 50 Passed
3. **RAG Grounding & Evaluation (Test-Set Suite)**: 5 / 5 Passed

The Postman collection is fully packaged in `postman/CARIVIX_AI.postman_collection.json` with environment settings in `postman/CARIVIX_AI.postman_environment.json`, ready for ongoing team use and CI/CD automation.
