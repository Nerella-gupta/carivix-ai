#!/usr/bin/env python3
"""
CARIVIX AI - Postman Collection Automated Test Runner
======================================================
Executes all endpoints defined in the Postman collection against the live
FastAPI server, validates HTTP status codes, latencies, response schemas,
and functional behaviors, and compiles an evidence report.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Any, Dict, List

import requests

BASE_URL = "http://127.0.0.1:8000"

def run_test_suite() -> Dict[str, Any]:
    print("=" * 75)
    print("  CARIVIX AI - POSTMAN API TEST SUITE RUNNER")
    print(f"  Target Server: {BASE_URL}")
    print(f"  Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    test_cases = [
        # --- 1. Health & System Status ---
        {
            "category": "Health & System Status",
            "name": "Check Service Health",
            "method": "GET",
            "endpoint": "/health",
            "payload": None,
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("status") == "healthy"
                and r.json().get("service") == "CARIVIX AI Model Service"
                and r.json().get("models_loaded", 0) >= 1
            ),
        },
        {
            "category": "Health & System Status",
            "name": "List Available Models",
            "method": "GET",
            "endpoint": "/api/v1/models",
            "payload": None,
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and isinstance(r.json().get("models"), list)
                and len(r.json()["models"]) >= 1
                and r.json()["models"][0].get("status") == "loaded"
            ),
        },
        {
            "category": "Health & System Status",
            "name": "Get Default Model Info",
            "method": "GET",
            "endpoint": "/api/v1/model/info",
            "payload": None,
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("status") == "loaded"
                and "algorithm" in r.json()
            ),
        },
        # --- 2. ML Inference Endpoints ---
        {
            "category": "ML Inference",
            "name": "Single Record Prediction (Default Model)",
            "method": "POST",
            "endpoint": "/api/v1/predict",
            "payload": {
                "age": 43,
                "income": 67976.67,
                "credit_score": 694,
                "loan_amount": 14857.28,
                "years_employed": 19,
                "education": "Master",
                "employment_status": "Employed",
                "marital_status": "Married",
                "housing_type": "Rent",
                "application_date": "2024-09-21",
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and "prediction" in r.json()
                and "model" in r.json()
            ),
        },
        {
            "category": "ML Inference",
            "name": "Single Record Prediction (Targeted RandomForest)",
            "method": "POST",
            "endpoint": "/api/v1/predict",
            "payload": {
                "age": 35,
                "income": 85000.00,
                "credit_score": 750,
                "loan_amount": 20000.00,
                "years_employed": 8,
                "education": "Bachelor",
                "employment_status": "Employed",
                "marital_status": "Single",
                "housing_type": "Own",
                "application_date": "2024-09-21",
                "model": "RandomForest",
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and "RandomForest" in r.json().get("model", "")
            ),
        },
        {
            "category": "ML Inference",
            "name": "Batch Prediction (Multiple Records)",
            "method": "POST",
            "endpoint": "/api/v1/predict/batch",
            "payload": {
                "records": [
                    {
                        "age": 43,
                        "income": 67976.67,
                        "credit_score": 694,
                        "loan_amount": 14857.28,
                        "years_employed": 19,
                        "education": "Master",
                        "employment_status": "Employed",
                        "marital_status": "Married",
                        "housing_type": "Rent",
                        "application_date": "2024-09-21",
                    },
                    {
                        "age": 29,
                        "income": 45000.00,
                        "credit_score": 610,
                        "loan_amount": 8000.00,
                        "years_employed": 3,
                        "education": "Bachelor",
                        "employment_status": "Employed",
                        "marital_status": "Single",
                        "housing_type": "Rent",
                        "application_date": "2024-09-21",
                    },
                ]
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and r.json().get("count") == 2
                and len(r.json().get("predictions", [])) == 2
            ),
        },
        # --- 3. AI Integration & RAG Endpoints ---
        {
            "category": "AI Integration & RAG",
            "name": "AI Query - RAG Route (Direct Factual)",
            "method": "POST",
            "endpoint": "/api/v1/ai/query",
            "payload": {
                "query": "What is CARIVIX AI?",
                "k": 3,
                "max_tokens": 256,
                "temperature": 0.0,
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and r.json().get("selected_route") == "rag"
                and len(r.json().get("rag_context", {}).get("retrieved_chunks", [])) >= 1
                and "CARIVIX" in (r.json().get("final_response") or "")
            ),
        },
        {
            "category": "AI Integration & RAG",
            "name": "AI Query - RAG Route (Supported ML Algorithms)",
            "method": "POST",
            "endpoint": "/api/v1/ai/query",
            "payload": {
                "query": "Which machine learning algorithms are supported?",
                "k": 3,
                "max_tokens": 256,
                "temperature": 0.0,
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and r.json().get("selected_route") == "rag"
                and len(r.json().get("rag_context", {}).get("retrieved_chunks", [])) >= 1
            ),
        },
        {
            "category": "AI Integration & RAG",
            "name": "AI Query - ML Route Intent (Feature Gap Handling)",
            "method": "POST",
            "endpoint": "/api/v1/ai/query",
            "payload": {
                "query": "Predict the customer churn probability for this account",
                "k": 3,
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and r.json().get("selected_route") == "ml"
                and "ML routing selected" in (r.json().get("final_response") or "")
            ),
        },
        {
            "category": "AI Integration & RAG",
            "name": "AI Query - Combined Route (Economic Forecasting)",
            "method": "POST",
            "endpoint": "/api/v1/ai/query",
            "payload": {
                "query": "Explain economic forecasting",
                "k": 3,
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and r.json().get("selected_route") == "combined"
                and r.json().get("rag_context") is not None
            ),
        },
        {
            "category": "AI Integration & RAG",
            "name": "AI Query - Out-of-Domain Refusal",
            "method": "POST",
            "endpoint": "/api/v1/ai/query",
            "payload": {
                "query": "What is the capital of France?",
                "k": 3,
            },
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and r.json().get("success") is True
                and any(
                    phrase in (r.json().get("final_response") or "").lower()
                    for phrase in ["not found", "information not", "does not contain"]
                )
            ),
        },
        # --- 4. Error Handling & Validation ---
        {
            "category": "Validation & Errors",
            "name": "Validation Error - Empty Query (HTTP 400)",
            "method": "POST",
            "endpoint": "/api/v1/ai/query",
            "payload": {"query": "   "},
            "expected_status": 400,
            "validate": lambda r: (
                r.status_code == 400
                and r.json().get("detail") == "Empty query"
            ),
        },
        {
            "category": "Validation & Errors",
            "name": "Validation Error - Missing Fields (HTTP 422)",
            "method": "POST",
            "endpoint": "/api/v1/predict",
            "payload": {"age": -5, "income": "invalid"},
            "expected_status": 422,
            "validate": lambda r: (
                r.status_code == 422
                and r.json().get("success") is False
            ),
        },
        # --- 5. Documentation & Schema Endpoints ---
        {
            "category": "Documentation & Schema",
            "name": "Root Redirect",
            "method": "GET",
            "endpoint": "/",
            "payload": None,
            "expected_status": 200,  # Following redirect leads to /docs 200
            "validate": lambda r: r.status_code in [200, 307],
        },
        {
            "category": "Documentation & Schema",
            "name": "OpenAPI Schema Spec",
            "method": "GET",
            "endpoint": "/openapi.json",
            "payload": None,
            "expected_status": 200,
            "validate": lambda r: (
                r.status_code == 200
                and "openapi" in r.json()
                and r.json().get("info", {}).get("title") == "CARIVIX AI Model Service"
            ),
        },
    ]

    results = []
    passed_count = 0
    failed_count = 0

    for idx, tc in enumerate(test_cases, start=1):
        url = f"{BASE_URL}{tc['endpoint']}"
        t0 = time.perf_counter()
        try:
            if tc["method"] == "GET":
                resp = requests.get(url, timeout=30)
            elif tc["method"] == "POST":
                resp = requests.post(url, json=tc["payload"], timeout=30)
            else:
                resp = requests.request(tc["method"], url, timeout=30)
            elapsed = time.perf_counter() - t0

            status_ok = (resp.status_code == tc["expected_status"])
            valid_ok = tc["validate"](resp)
            test_passed = status_ok and valid_ok

            if test_passed:
                passed_count += 1
                status_label = "[PASSED]"
            else:
                failed_count += 1
                status_label = "[FAILED]"

            try:
                resp_data = resp.json()
            except Exception:
                resp_data = resp.text[:200]

            record = {
                "id": idx,
                "category": tc["category"],
                "name": tc["name"],
                "method": tc["method"],
                "endpoint": tc["endpoint"],
                "expected_status": tc["expected_status"],
                "actual_status": resp.status_code,
                "latency_sec": round(elapsed, 4),
                "passed": test_passed,
                "response_sample": resp_data,
            }
            results.append(record)

            print(f"[{idx:02d}/{len(test_cases):02d}] {status_label} {tc['method']:<4} {tc['endpoint']:<25} | "
                  f"Status: {resp.status_code} (Exp: {tc['expected_status']}) | "
                  f"Time: {elapsed:.4f}s | {tc['name']}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            failed_count += 1
            print(f"[{idx:02d}/{len(test_cases):02d}] [FAILED] {tc['method']:<4} {tc['endpoint']:<25} | "
                  f"ERROR: {exc} | Time: {elapsed:.4f}s | {tc['name']}")
            results.append({
                "id": idx,
                "category": tc["category"],
                "name": tc["name"],
                "method": tc["method"],
                "endpoint": tc["endpoint"],
                "expected_status": tc["expected_status"],
                "actual_status": "ERROR",
                "latency_sec": round(elapsed, 4),
                "passed": False,
                "error": str(exc),
            })

    print("-" * 75)
    print(f"  TEST EXECUTION SUMMARY: {passed_count}/{len(test_cases)} Passed ({passed_count/len(test_cases)*100:.1f}%) | "
          f"{failed_count} Failed")
    print("=" * 75)

    os.makedirs("reports", exist_ok=True)
    report_file = os.path.join("reports", "postman_api_test_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "target": BASE_URL,
            "total_tests": len(test_cases),
            "passed": passed_count,
            "failed": failed_count,
            "results": results,
        }, f, indent=2)

    print(f"  Results saved to: {report_file}\n")
    return {"total": len(test_cases), "passed": passed_count, "failed": failed_count, "results": results}

if __name__ == "__main__":
    run_test_suite()
