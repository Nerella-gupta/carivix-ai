# CARIVIX AI - RAG Pipeline Evaluation & Optimization Report

_Generated: 2026-08-04 11:35:43_

---

## 1. Current System Configuration

| Component | Configuration |
|---|---|
| Document Loader | Implemented (PDF/DOCX/TXT/CSV) |
| Text Preprocessing | Enabled |
| Chunk Size | 500 |
| Chunk Overlap | 50 |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Vector Database | FAISS |
| Retrieval Top-k | 5 |
| LLM Runtime | Ollama |
| Processing Device | CPU |


## 2. Vector Database Evaluation

### FAISS Index Validation

| Property | Value |
|---|---|
| FAISS Index Loaded | True |
| Status | PASS |
| Vector Count | 12 |
| Document Count | 12 |
| Embedding Dimension | 384 |
| Expected Dimension | 384 |
| Dimension Match | True |
| Index Load Time | 0.009s |
| Index Path | `D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training\data\vector_store\index.faiss` |
| Metadata Path | `D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training\data\vector_store\index.pkl` |

### Similarity Search Summary

| Query | Results | Avg Score | Lexical Rel. | Duplicates | Retrieval |
|---|---|---|---|---|---|
| What is CARIVIX AI? | 5 | 0.5231 | 0.6000 | 0 | 8.184s |
| What machine learning algorithms are supported? | 5 | 0.5127 | 0.3000 | 0 | 0.049s |
| How does the platform handle data preprocessing? | 5 | 0.3995 | 0.4500 | 0 | 0.045s |
| What is economic analysis? | 5 | 0.4826 | 0.7000 | 0 | 0.045s |
| Tell me about market prediction. | 5 | 0.4692 | 0.2000 | 0 | 0.039s |

### Detailed Retrieved Documents

| Query | Rank | Chunk ID | Source | Score | Text Preview |
|---|---|---|---|---|---|
| What is CARIVIX AI? | 1 | 1 | sample_about_carivix.txt | 0.7176 | CARIVIX AI Platform Overview =============================  ... |
| What is CARIVIX AI? | 2 | 3 | sample_about_carivix.txt | 0.5811 | 3. Model Training: CARIVIX AI supports multiple ML algorithm... |
| What is CARIVIX AI? | 3 | 2 | sample_about_carivix.txt | 0.5691 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses ... |
| What is CARIVIX AI? | 4 | 5 | sample_economics.txt | 0.3748 | Machine Learning in Finance --------------------------- Mach... |
| What is CARIVIX AI? | 5 | 5 | sample_about_carivix.txt | 0.3729 | The platform is built with Python 3.11 and follows productio... |
| What machine learning algorithms are sup | 1 | 3 | sample_about_carivix.txt | 0.5692 | 3. Model Training: CARIVIX AI supports multiple ML algorithm... |
| What machine learning algorithms are sup | 2 | 2 | sample_about_carivix.txt | 0.5409 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses ... |
| What machine learning algorithms are sup | 3 | 5 | sample_economics.txt | 0.5122 | Machine Learning in Finance --------------------------- Mach... |
| What machine learning algorithms are sup | 4 | 4 | sample_about_carivix.txt | 0.4782 | 5. Model Evaluation: Comprehensive evaluation metrics for bo... |
| What machine learning algorithms are sup | 5 | 6 | sample_economics.txt | 0.4632 | Time series forecasting uses models like ARIMA, Prophet, and... |
| How does the platform handle data prepro | 1 | 2 | sample_about_carivix.txt | 0.4246 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses ... |
| How does the platform handle data prepro | 2 | 5 | sample_about_carivix.txt | 0.4142 | The platform is built with Python 3.11 and follows productio... |
| How does the platform handle data prepro | 3 | 1 | sample_about_carivix.txt | 0.3927 | CARIVIX AI Platform Overview =============================  ... |
| How does the platform handle data prepro | 4 | 4 | sample_economics.txt | 0.3869 | Technical analysis uses historical price and volume data to ... |
| How does the platform handle data prepro | 5 | 4 | sample_about_carivix.txt | 0.3791 | 5. Model Evaluation: Comprehensive evaluation metrics for bo... |
| What is economic analysis? | 1 | 1 | sample_economics.txt | 0.5469 | Economic Analysis and Market Trends ========================... |
| What is economic analysis? | 2 | 4 | sample_economics.txt | 0.5377 | Technical analysis uses historical price and volume data to ... |
| What is economic analysis? | 3 | 3 | sample_economics.txt | 0.4544 | Market Analysis --------------- Stock markets are influenced... |
| What is economic analysis? | 4 | 2 | sample_about_carivix.txt | 0.4422 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses ... |
| What is economic analysis? | 5 | 1 | sample_about_carivix.txt | 0.4316 | CARIVIX AI Platform Overview =============================  ... |
| Tell me about market prediction. | 1 | 3 | sample_economics.txt | 0.5219 | Market Analysis --------------- Stock markets are influenced... |
| Tell me about market prediction. | 2 | 6 | sample_economics.txt | 0.4681 | Time series forecasting uses models like ARIMA, Prophet, and... |
| Tell me about market prediction. | 3 | 5 | sample_economics.txt | 0.4586 | Machine Learning in Finance --------------------------- Mach... |
| Tell me about market prediction. | 4 | 2 | sample_about_carivix.txt | 0.4568 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses ... |
| Tell me about market prediction. | 5 | 7 | sample_economics.txt | 0.4408 | Risk Management --------------- Value at Risk (VaR) measures... |

**Overall Vector DB Status: `PASS`**


## 3. Retrieval Performance Metrics

| Config | Chunks | Embed Time | Index Time | Retrieval | Total Query |
|---|---|---|---|---|---|
| **300/50/k=3** | 22 | 0.960s | 0.001s | 0.038s | 0.038s |
| **500/30/k=3** | 12 | 0.718s | 0.001s | 0.044s | 0.044s |
| **500/50/k=3** | 12 | 0.654s | 0.001s | 0.044s | 0.044s |
| **800/30/k=3** | 7 | 0.528s | 0.002s | 0.042s | 0.042s |
| **300/100/k=3** | 22 | 0.898s | 0.001s | 0.052s | 0.052s |
| **500/100/k=3** | 12 | 0.740s | 0.001s | 0.055s | 0.055s |
| **300/30/k=5** | 22 | 0.839s | 0.001s | 0.044s | 0.044s |
| **300/50/k=5** | 22 | 0.841s | 0.001s | 0.056s | 0.056s |
| **500/50/k=5** | 12 | 0.619s | 0.001s | 0.045s | 0.045s |
| **500/30/k=5** | 12 | 0.639s | 0.001s | 0.045s | 0.045s |
| **300/100/k=5** | 22 | 0.884s | 0.002s | 0.064s | 0.064s |
| **300/30/k=10** | 22 | 0.912s | 0.001s | 0.040s | 0.040s |
| **500/30/k=10** | 12 | 0.606s | 0.003s | 0.043s | 0.043s |
| **800/50/k=5** | 7 | 0.735s | 0.001s | 0.060s | 0.060s |
| **800/50/k=10** | 7 | 0.663s | 0.001s | 0.052s | 0.052s |
| **300/100/k=10** | 22 | 0.905s | 0.001s | 0.059s | 0.059s |
| **800/100/k=3** | 7 | 1.270s | 0.012s | 0.093s | 0.093s |
| **500/100/k=10** | 12 | 0.824s | 0.001s | 0.055s | 0.055s |
| **800/100/k=5** | 7 | 0.629s | 0.001s | 0.071s | 0.071s |
| **800/30/k=10** | 7 | 1.116s | 0.001s | 0.058s | 0.058s |
| **800/30/k=5** | 7 | 0.720s | 0.001s | 0.077s | 0.077s |
| **300/30/k=3** | 22 | 8.116s | 0.004s | 0.117s | 0.117s |
| **300/50/k=10** | 22 | 0.943s | 0.012s | 0.083s | 0.083s |
| **500/100/k=5** | 12 | 1.062s | 0.002s | 0.106s | 0.106s |
| **500/50/k=10** | 12 | 0.672s | 0.001s | 0.107s | 0.107s |
| **800/100/k=10** | 7 | 1.093s | 0.018s | 0.127s | 0.127s |
| **800/50/k=3** | 7 | 0.567s | 0.001s | 0.179s | 0.179s |


## 4. Latency Analysis

### Average Metrics Across All Configurations

| Metric | Average |
|---|---|
| Embedding Generation (batch) | 1.080s |
| FAISS Indexing (batch)   | 0.003s |
| Retrieval Latency (per query) | 0.069s |
| Average Similarity Score | 0.4763 |

### Fastest vs Slowest Configuration

| Metric | Fastest | Slowest |
|---|---|---|
| Config | 300/50/k=3 | 800/50/k=3 |
| Retrieval Latency | 0.038s | 0.179s |
| Average Similarity | 0.5510 | 0.4696 |


## 5. Similarity Search Results

### Query: What is CARIVIX AI?

| Rank | Chunk ID | Source | Score | Text |
|---|---|---|---|---|
| 1 | 1 | sample_about_carivix.txt | 0.7176 | CARIVIX AI Platform Overview =============================  CARIVIX AI is a comp... |
| 2 | 3 | sample_about_carivix.txt | 0.5811 | 3. Model Training: CARIVIX AI supports multiple ML algorithms including Random F... |
| 3 | 2 | sample_about_carivix.txt | 0.5691 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses advanced machine lea... |
| 4 | 5 | sample_economics.txt | 0.3748 | Machine Learning in Finance --------------------------- Machine learning has rev... |
| 5 | 5 | sample_about_carivix.txt | 0.3729 | The platform is built with Python 3.11 and follows production-grade software eng... |

### Query: What machine learning algorithms are supported?

| Rank | Chunk ID | Source | Score | Text |
|---|---|---|---|---|
| 1 | 3 | sample_about_carivix.txt | 0.5692 | 3. Model Training: CARIVIX AI supports multiple ML algorithms including Random F... |
| 2 | 2 | sample_about_carivix.txt | 0.5409 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses advanced machine lea... |
| 3 | 5 | sample_economics.txt | 0.5122 | Machine Learning in Finance --------------------------- Machine learning has rev... |
| 4 | 4 | sample_about_carivix.txt | 0.4782 | 5. Model Evaluation: Comprehensive evaluation metrics for both classification (a... |
| 5 | 6 | sample_economics.txt | 0.4632 | Time series forecasting uses models like ARIMA, Prophet, and LSTM networks to pr... |

### Query: How does the platform handle data preprocessing?

| Rank | Chunk ID | Source | Score | Text |
|---|---|---|---|---|
| 1 | 2 | sample_about_carivix.txt | 0.4246 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses advanced machine lea... |
| 2 | 5 | sample_about_carivix.txt | 0.4142 | The platform is built with Python 3.11 and follows production-grade software eng... |
| 3 | 1 | sample_about_carivix.txt | 0.3927 | CARIVIX AI Platform Overview =============================  CARIVIX AI is a comp... |
| 4 | 4 | sample_economics.txt | 0.3869 | Technical analysis uses historical price and volume data to predict future price... |
| 5 | 4 | sample_about_carivix.txt | 0.3791 | 5. Model Evaluation: Comprehensive evaluation metrics for both classification (a... |

### Query: What is economic analysis?

| Rank | Chunk ID | Source | Score | Text |
|---|---|---|---|---|
| 1 | 1 | sample_economics.txt | 0.5469 | Economic Analysis and Market Trends ====================================  Macroe... |
| 2 | 4 | sample_economics.txt | 0.5377 | Technical analysis uses historical price and volume data to predict future price... |
| 3 | 3 | sample_economics.txt | 0.4544 | Market Analysis --------------- Stock markets are influenced by multiple factors... |
| 4 | 2 | sample_about_carivix.txt | 0.4422 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses advanced machine lea... |
| 5 | 1 | sample_about_carivix.txt | 0.4316 | CARIVIX AI Platform Overview =============================  CARIVIX AI is a comp... |

### Query: Tell me about market prediction.

| Rank | Chunk ID | Source | Score | Text |
|---|---|---|---|---|
| 1 | 3 | sample_economics.txt | 0.5219 | Market Analysis --------------- Stock markets are influenced by multiple factors... |
| 2 | 6 | sample_economics.txt | 0.4681 | Time series forecasting uses models like ARIMA, Prophet, and LSTM networks to pr... |
| 3 | 5 | sample_economics.txt | 0.4586 | Machine Learning in Finance --------------------------- Machine learning has rev... |
| 4 | 2 | sample_about_carivix.txt | 0.4568 | Core Capabilities: 1. Economic Forecasting: CARIVIX AI uses advanced machine lea... |
| 5 | 7 | sample_economics.txt | 0.4408 | Risk Management --------------- Value at Risk (VaR) measures the potential loss ... |


## 6. Optimization Comparison Table

| Rank | Chunk Size | Overlap | Top-k | Avg Similarity | Lexical Rel. | Retrieval | Total Query | Composite |
|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 50 | 3 | 0.5510 | 0.5500 | 0.038s | 0.038s | 0.6455 |
| 2 | 500 | 30 | 3 | 0.5139 | 0.5833 | 0.044s | 0.044s | 0.6310 |
| 3 | 500 | 50 | 3 | 0.5139 | 0.5833 | 0.044s | 0.044s | 0.6303 |
| 4 | 800 | 30 | 3 | 0.4696 | 0.5667 | 0.042s | 0.042s | 0.6171 |
| 5 | 300 | 100 | 3 | 0.5510 | 0.5500 | 0.052s | 0.052s | 0.6131 |
| 6 | 500 | 100 | 3 | 0.5139 | 0.5833 | 0.055s | 0.055s | 0.6053 |
| 7 | 300 | 30 | 5 | 0.5167 | 0.4900 | 0.044s | 0.044s | 0.6035 |
| 8 | 300 | 50 | 5 | 0.5167 | 0.4900 | 0.056s | 0.056s | 0.5773 |
| 9 | 500 | 50 | 5 | 0.4774 | 0.4500 | 0.045s | 0.045s | 0.5769 |
| 10 | 500 | 30 | 5 | 0.4774 | 0.4500 | 0.045s | 0.045s | 0.5765 |
| 11 | 300 | 100 | 5 | 0.5159 | 0.4900 | 0.064s | 0.064s | 0.5577 |
| 12 | 300 | 30 | 10 | 0.4676 | 0.3250 | 0.040s | 0.040s | 0.5474 |
| 13 | 500 | 30 | 10 | 0.4325 | 0.3100 | 0.043s | 0.043s | 0.5266 |
| 14 | 800 | 50 | 5 | 0.4413 | 0.4100 | 0.060s | 0.060s | 0.5207 |
| 15 | 800 | 50 | 10 | 0.4170 | 0.3214 | 0.052s | 0.052s | 0.5056 |
| 16 | 300 | 100 | 10 | 0.4672 | 0.3250 | 0.059s | 0.059s | 0.5052 |
| 17 | 800 | 100 | 3 | 0.4696 | 0.5667 | 0.093s | 0.093s | 0.5026 |
| 18 | 500 | 100 | 10 | 0.4325 | 0.3100 | 0.055s | 0.055s | 0.4989 |
| 19 | 800 | 100 | 5 | 0.4413 | 0.4100 | 0.071s | 0.071s | 0.4964 |
| 20 | 800 | 30 | 10 | 0.4170 | 0.3214 | 0.058s | 0.058s | 0.4925 |
| 21 | 800 | 30 | 5 | 0.4413 | 0.4100 | 0.077s | 0.077s | 0.4825 |
| 22 | 300 | 30 | 3 | 0.5510 | 0.5500 | 0.117s | 0.117s | 0.4694 |
| 23 | 300 | 50 | 10 | 0.4676 | 0.3250 | 0.083s | 0.083s | 0.4517 |
| 24 | 500 | 100 | 5 | 0.4774 | 0.4500 | 0.106s | 0.106s | 0.4420 |
| 25 | 500 | 50 | 10 | 0.4325 | 0.3100 | 0.107s | 0.107s | 0.3836 |
| 26 | 800 | 100 | 10 | 0.4170 | 0.3214 | 0.127s | 0.127s | 0.3379 |
| 27 | 800 | 50 | 3 | 0.4696 | 0.5667 | 0.179s | 0.179s | 0.3109 |


## 7. Best Configuration

| Parameter | Recommended Value |
|---|---|
| Chunk Size | 300 |
| Chunk Overlap | 50 |
| Retrieval Top-k | 3 |
| Number of Chunks | 22 |
| Embedding Time | 0.960s |
| FAISS Indexing Time | 0.001s |
| Retrieval Latency | 0.038s |
| Total Query Time | 0.038s |
| Average Similarity | 0.5510 |
| Lexical Relevance | 0.5500 |
| Composite Score | 0.6455 |


## 8. Observations

- The FAISS index loads successfully with 12 vectors at dimension 384.
- No duplicate retrieval results were detected across sample queries.
- The fastest retrieval config is 300/50/k=3 (0.038s/query).
- The highest-average-similarity config is 300/50/k=3 (avg similarity 0.5510).
- Compared to the current default (500/50/k=5), the best config changes retrieval latency by -0.0074s and average similarity by +0.0736.


## 9. Recommendations

1. **Adopt the recommended configuration** (chunk_size=300, chunk_overlap=50, top_k=3) based on its composite score balancing retrieval speed and relevance.
2. **Apply score-based filtering** (e.g., drop chunks below a similarity threshold) to reduce noise in the retrieved context.
3. **Consider re-indexing the production vector store** with the recommended chunking parameters for a measurable retrieval quality gain.
4. **Use batched embedding generation** and persist the sentence-transformer model locally to reduce cold-start latency.
5. **Periodically re-run this evaluation** (evaluate_rag.py) when documents or the embedding model change.


## 10. Conclusion

The CARIVIX AI RAG pipeline's vector database integration is **PASS**.

Based on a grid search over chunk size, chunk overlap, and top-k, the recommended configuration is **chunk_size=300, chunk_overlap=50, top_k=3** with an average similarity of 0.5510 and a retrieval latency of 0.038s.

These recommendations are intended as a starting point; adopt them incrementally and validate against real user queries.
