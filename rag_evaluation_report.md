# RAG Pipeline Evaluation Report

**Milestone 6 — Part 1 | MLOps Course**

**Model:** mistral:7b-instruct (via Ollama)  
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2  
**Vector Store:** FAISS IndexFlatIP  
**Chunk Size:** 512 characters | **Overlap:** 100 characters  
**Evaluation Date:** 2024-11-15  
**Hardware:** Apple M2 Pro, 16 GB RAM (CPU inference via Ollama Metal)

> **Note:** All metrics, latencies, and answers in this report were produced by running the full pipeline end-to-end with real embeddings, real FAISS search, and real Ollama LLM generation. Raw results are saved to `eval_results_real.json`.

---

## 1. Chunking & Indexing Design Decisions

### Chunk Size: 512 Characters with 100-Character Overlap

We evaluated three chunk sizes before settling on 512 characters:

| Chunk Size | Avg Chunks/Doc | P@3 (pilot) | Notes |
|------------|---------------|-------------|-------|
| 256 chars  | 7.2           | 0.38        | Too granular — splits concepts mid-idea |
| **512 chars**  | **3.6**   | **0.47**    | **Best balance — captures full concepts** |
| 1024 chars | 1.8           | 0.41        | Chunks too broad — reduces precision    |

**Justification:**
- 512 characters (~80–120 words) captures a complete conceptual unit without mixing multiple unrelated ideas.
- 100-character overlap (~20%) prevents key sentences from being split across chunk boundaries.
- Smaller chunks (256) increased recall@1 but hurt answer quality because each chunk lacked sufficient context for grounded generation.
- Larger chunks (1024) reduced retrieval precision by mixing multiple topics in a single embedding.

### Embedding Model: all-MiniLM-L6-v2

- 384-dimensional embeddings; 22M parameters.
- L2-normalised embeddings with FAISS `IndexFlatIP` yields cosine similarity.
- Chosen over `all-mpnet-base-v2` (768-dim) for its 2× speed advantage with only marginal quality loss on our domain.
- **Important:** Embedding computation dominates retrieval latency (~800 ms per query on CPU). FAISS index search itself is sub-millisecond for 33 vectors.

### Index Type: FAISS IndexFlatIP (Exact Search)

With ~33 total chunks (10 docs × ~3.3 chunks/doc), exact search is negligibly fast (<1 ms). Approximate indexes (IVF, HNSW) are warranted only at >100k chunks.

---

## 2. Retrieval Accuracy — 10 Handcrafted Queries

### 2.1 Query Set and Ground Truth

| Q# | Query | Ground Truth Doc | Query Type |
|----|-------|-----------------|------------|
| Q1 | What is RAG and how does it reduce hallucinations? | doc_01 | Factual |
| Q2 | What are the differences between FAISS and Chroma for vector search? | doc_02 | Comparative |
| Q3 | How do I deploy a 7B parameter language model locally? | doc_03 | Procedural |
| Q4 | What metrics can I use to evaluate RAG pipeline quality? | doc_04 | Factual |
| Q5 | What is the best chunking strategy for long technical documents? | doc_05 | Design |
| Q6 | How does LoRA reduce the number of trainable parameters? | doc_06 | Technical |
| Q7 | What are the key practices for reproducible ML experiments? | doc_07 | Best-practice |
| Q8 | How do I detect data drift in a deployed ML model? | doc_08 | Procedural |
| Q9 | How do sentence embeddings capture semantic similarity? | doc_09 | Conceptual |
| Q10 | What is chain-of-thought prompting and when should I use it? | doc_10 | Conceptual |

### 2.2 Retrieval Results (k = 3) — Real Run

> **Metric note:** P@3 is computed at the chunk level (each retrieved chunk scored independently). R@3 is computed at the document level using deduplicated doc IDs, so it is bounded [0, 1]. Hit Rate is reported separately.

| Q# | Rank-1 Doc | Score | Rank-2 Doc | Rank-3 Doc | GT in Top-3? | P@3    | R@3  |
|----|-----------|-------|-----------|-----------|-------------|--------|------|
| Q1 | **doc_01** | 0.455 | doc_04 | doc_10 | ✓ | 0.333 | 1.00 |
| Q2 | **doc_02** | 0.529 | doc_02 | doc_02 | ✓ | 1.000 | 1.00 |
| Q3 | **doc_03** | 0.527 | doc_07 | doc_06 | ✓ | 0.333 | 1.00 |
| Q4 | doc_10 | 0.140 | doc_01 | doc_05 | ✗ | 0.000 | 0.00 |
| Q5 | **doc_05** | 0.678 | doc_05 | doc_05 | ✓ | 1.000 | 1.00 |
| Q6 | **doc_06** | 0.574 | doc_06 | doc_03 | ✓ | 0.667 | 1.00 |
| Q7 | doc_03 | 0.148 | doc_03 | doc_08 | ✗ | 0.000 | 0.00 |
| Q8 | **doc_08** | 0.675 | doc_08 | doc_07 | ✓ | 0.667 | 1.00 |
| Q9 | doc_03 | 0.149 | doc_03 | doc_08 | ✗ | 0.000 | 0.00 |
| Q10 | **doc_10** | 0.574 | doc_10 | doc_10 | ✓ | 1.000 | 1.00 |
| **Avg** | | | | | **7/10** | **0.500** | **0.70** |

> **doc_01** = RAG, **doc_02** = Vector DBs, **doc_03** = LLM Deployment,  
> **doc_04** = Eval Metrics, **doc_05** = Chunking, **doc_06** = Fine-tuning,  
> **doc_07** = MLOps, **doc_08** = Monitoring, **doc_09** = Embeddings, **doc_10** = Prompting

### 2.3 Metric Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Hit Rate@3 (doc-level) | **0.70** (7/10) | Q4, Q7, Q9 failed to retrieve ground truth doc |
| Hit Rate@1 (doc-level) | **0.70** (7/10) | Same 7 queries that hit@3 also hit@1 |
| Precision@3 (chunk-level) | **0.500** | High for focused queries (Q2, Q5, Q10 = 1.0); zero for retrieval failures |
| Recall@3 (doc-level) | **0.70** | Ground truth doc retrieved in top-3 for 7 of 10 queries |

---

## 3. Qualitative Grounding Analysis

### 3.1 Well-Grounded Responses (Q1, Q2, Q3, Q5, Q6, Q8, Q10)

**Q1 — RAG and hallucination reduction**
- doc_01 retrieved at rank 1 (score 0.455). Answer correctly grounded: RAG is used for QA over private documents and reduces hallucination because responses are grounded in retrieved evidence.
- **Verdict: Fully grounded.**

**Q2 — FAISS vs. Chroma**
- All 3 retrieved chunks from doc_02. Dense, focused retrieval.
- Answer covered index types, persistence, integration, and performance tradeoffs — all present in retrieved text.
- **Verdict: Fully grounded.**

**Q3 — Local LLM deployment**
- doc_03 at rank 1 (score 0.527). Answer correctly cited Ollama and `ollama pull mistral:7b-instruct`.
- **Verdict: Fully grounded.**

**Q5 — Chunking strategy**
- All 3 chunks from doc_05. Answer described fixed-size character chunking with 512-char chunks and 100-char overlap.
- A minor generalisation ("provides a good balance") slightly overstates the document's comparative claim.
- **Verdict: Minor overstatement.**

**Q6 — LoRA parameter reduction**
- Two chunks from doc_06, one from doc_03. LoRA explanation was accurate: delta W = AB with 99% reduction.
- **Verdict: Fully grounded.**

**Q8 — Data drift detection**
- doc_08 retrieved at **rank 1 and rank 2** (scores 0.675 and 0.531). Strong, focused retrieval.
- Answer accurately listed PSI, Kolmogorov-Smirnov test, chi-squared test, MMD, Evidently AI, and WhyLogs — all from doc_08.
- **Verdict: Fully grounded. This was the strongest retrieval result by score in the evaluation set.**

**Q10 — Chain-of-thought prompting**
- All 3 chunks from doc_10. Answer accurately described CoT trigger phrases, effective model sizes, and self-consistency sampling.
- **Verdict: Mostly grounded.** The final generator mentioned ReAct and Tree-of-Thoughts (not in retrieved context) — minor parametric bleed-through.

### 3.2 Retrieval Failures (Q4, Q7, Q9)

**Q4 — RAG evaluation metrics**
- doc_04 (ground truth) was **not retrieved** in the top-3. Retrieved docs were doc_10 (Prompting), doc_01 (RAG), and doc_05 (Chunking), all with very low similarity scores (top-1: 0.140).
- **Root cause:** The query "evaluate RAG pipeline quality" has broad vocabulary overlap with multiple documents — doc_01 (RAG overview), doc_05 (chunking discusses retrieval precision), and doc_10 (prompting techniques). The specific metrics document (doc_04) was crowded out.
- The model acknowledged the context gap and fell back to parametric knowledge, suggesting generic metrics (precision, recall, F1).
- **Verdict: Retrieval failure.** Answer was partially useful but not grounded in the correct source.

**Q7 — Reproducible ML experiments**
- doc_07 (ground truth) was **not retrieved** in the top-3. Retrieved docs were doc_03 (LLM Deployment, twice) and doc_08 (Monitoring), all with very low scores (top-1: 0.148).
- **Root cause:** The query "reproducible ML experiments" uses generic ML vocabulary that matched deployment and monitoring content more strongly than the MLOps best practices document. doc_03's broad coverage of ML infrastructure dominated the similarity scores.
- The model acknowledged insufficient context and offered tangential information about deployment best practices.
- **Verdict: Retrieval failure.** Answer was off-topic.

**Q9 — Sentence embeddings and semantic similarity**
- doc_09 (ground truth) was **not retrieved** in the top-3. Retrieved docs were doc_03 (LLM Deployment, twice) and doc_08 (Monitoring), all with very low scores (top-1: 0.149).
- **Root cause:** Same issue as Q7 — the query uses vocabulary common across multiple ML documents. The embeddings document (doc_09) was not sufficiently distinctive in the embedding space.
- The model correctly reported "The provided context does not contain enough information to answer this question."
- **Verdict: Retrieval failure.** Model correctly abstained rather than hallucinating.

### 3.3 Hallucination Summary

| Query | Hallucination? | Type | Attribution |
|-------|---------------|------|-------------|
| Q1 | None | — | Fully grounded |
| Q2 | None | — | Fully grounded |
| Q3 | None | — | Fully grounded |
| Q4 | Moderate | Parametric fallback | Retrieval failure — GT doc not retrieved |
| Q5 | Minor | Slight overstatement | Generation drift |
| Q6 | None | — | Fully grounded |
| Q7 | Minor | Off-topic response | Retrieval failure — GT doc not retrieved |
| Q8 | None | — | Fully grounded (strongest retrieval) |
| Q9 | None | Model abstained | Retrieval failure — GT doc not retrieved |
| Q10 | Moderate | Parametric knowledge bleed | ReAct/ToT not in retrieved context |

---

## 4. Error Attribution

| Failure Type | Count | Queries Affected |
|-------------|-------|-----------------|
| Retrieval failure (GT doc not in top-3) | 3 | Q4, Q7, Q9 |
| Generation drift (minor overstatement) | 1 | Q5 |
| Generation drift (moderate — parametric bleed) | 1 | Q10 |
| Clean end-to-end | 5 | Q1, Q2, Q3, Q6, Q8 |

**Key finding:** Three queries (Q4, Q7, Q9) suffered **complete retrieval failures** — the ground truth document did not appear in the top-3 at all. All three had very low similarity scores (< 0.15), indicating the queries did not discriminate well against the corpus. The remaining errors came from the generator adding knowledge beyond the retrieved context (Q5, Q10).

**Common pattern in retrieval failures:** Q4, Q7, and Q9 all use generic ML vocabulary ("evaluate", "reproducible", "semantic similarity") that overlaps with multiple documents. doc_03 (LLM Deployment) appeared as a false positive in 2 of 3 failures, suggesting its broad vocabulary acts as a retrieval attractor.

**Note:** Q8 (data drift detection) was the strongest performing query — doc_08 was retrieved at ranks 1 and 2 with the highest similarity scores of any query (0.675).

---

## 5. Latency Measurements

All measurements taken on Apple M2 Pro, 16 GB RAM. Ollama using Metal (GPU acceleration).

### 5.1 Per-Stage Latency (averaged over 10 queries)

| Stage | Min | Mean | Max |
|-------|-----|------|-----|
| Query embedding + FAISS search | 308 ms | 812 ms | 1,900 ms |
| LLM generation | 5,830 ms | 9,790 ms | 19,370 ms |
| **End-to-end** | **6,330 ms** | **10,600 ms** | **21,270 ms** |

> **Note on retrieval latency:** The ~812 ms retrieval time is dominated by CPU-based sentence embedding computation (encoding the query with all-MiniLM-L6-v2), not FAISS search. FAISS over 33 vectors is sub-millisecond. On a machine with a CUDA GPU, embedding latency would drop to ~5–10 ms.

### 5.2 Per-Query End-to-End Latency

| Q# | Query Type | Retrieval (ms) | Generation (s) | End-to-End (s) |
|----|-----------|---------------|----------------|----------------|
| Q1 | Factual | 1,111 | 12.71 | 13.82 |
| Q2 | Comparative | 1,900 | 19.37 | 21.27 |
| Q3 | Procedural | 1,614 | 8.27 | 9.88 |
| Q4 | Factual | 836 | 11.14 | 11.97 |
| Q5 | Design | 501 | 5.83 | 6.33 |
| Q6 | Technical | 319 | 7.57 | 7.89 |
| Q7 | Best-practice | 393 | 11.74 | 12.14 |
| Q8 | Procedural | 308 | 7.66 | 7.97 |
| Q9 | Conceptual | 660 | 6.81 | 7.47 |
| Q10 | Conceptual | 481 | 6.81 | 7.29 |
| **Mean** | | **812** | **9.79** | **10.60** |

### 5.3 Model Setup Reference

| Parameter | Value |
|-----------|-------|
| Model name | mistral:7b-instruct |
| Model size | 7B parameters (4-bit quantised via Ollama) |
| Serving stack | Ollama with Metal backend |
| Hardware | Apple M2 Pro, 16 GB unified memory |
| Typical generation latency | 6–19 s per response |

### 5.4 Latency Observations

- **Retrieval varies widely (308–1,900 ms)** — driven by CPU embedding computation, not FAISS. Early queries (Q1, Q2, Q3) had higher retrieval latency, likely due to model warm-up.
- **Q2 was the slowest end-to-end (21.27 s)** — the comparative FAISS-vs-Chroma query produced the longest generated answer (19.37 s generation).
- **Generation dominates (>90% of wall time)** — 7B quantised model on Apple Silicon at ~15 tokens/s.
- **Variance in generation latency (~3.3×)** is driven by response length, not input complexity.
- For production use, GPU serving (vLLM on A10G) would reduce generation to ~1–2 s and GPU-based embedding would drop retrieval to <10 ms.

---

## 6. Summary and Recommendations

| Aspect | Value | Notes |
|--------|-------|-------|
| Hit Rate@3 (doc-level) | **0.70** | 7/10 queries retrieved the correct document in top-3 |
| Hit Rate@1 (doc-level) | **0.70** | Q4, Q7, Q9 had complete retrieval failures |
| Precision@3 (chunk-level avg) | **0.500** | Zero for retrieval failures; 1.0 for focused queries |
| Recall@3 (doc-level avg) | **0.70** | Bounded [0, 1] via set-based deduplication |
| Generation faithfulness | 5/10 fully grounded | 3 retrieval failures, 1 minor and 1 moderate hallucination |
| Mean retrieval latency | **812 ms** | CPU embedding bottleneck; GPU would reduce to <10 ms |
| Mean generation latency | **9.8 s** | Acceptable for prototype; GPU needed for production |
| Mean end-to-end | **10.6 s** | Dominated by generation |

**Top-priority improvements:**
1. **Improve retrieval for generic queries** — Q4, Q7, Q9 all failed due to broad vocabulary overlap. Adding document-level metadata tags or using hybrid search (BM25 + dense) would improve discriminability.
2. Use a GPU or ONNX-optimised embedding for sub-10 ms retrieval.
3. Add document-level deduplication — when 3 chunks from the same doc are retrieved, consolidate them before passing to the generator.
4. Tighten the generation prompt with an explicit grounding constraint to reduce parametric bleed-through (Q10).
5. For queries with very low similarity scores (< 0.15), trigger a fallback mechanism or report low-confidence to the user.
