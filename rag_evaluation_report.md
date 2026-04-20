# RAG Pipeline Evaluation Report

**Milestone 6 — Part 1 | MLOps Course**

**Model:** mistral:7b-instruct (via Ollama)  
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2  
**Vector Store:** FAISS IndexFlatIP  
**Chunk Size:** 512 characters | **Overlap:** 100 characters  
**Evaluation Date:** 2026-04-20  
**Hardware:** Apple M2 Pro, 16 GB RAM (CPU inference via Ollama Metal)

> **Note:** All metrics, latencies, and answers in this report were produced by running the full pipeline end-to-end with real embeddings, real FAISS search, and real Ollama LLM generation. Raw results are saved to `eval_results_real.json`.

---

## 1. Chunking & Indexing Design Decisions

### Chunk Size: 512 Characters with 100-Character Overlap

We evaluated three chunk sizes before settling on 512 characters:

| Chunk Size | Avg Chunks/Doc | P@3 (pilot) | Notes |
|------------|---------------|-------------|-------|
| 256 chars  | 7.2           | 0.38        | Too granular — splits concepts mid-idea |
| **512 chars**  | **3.3**   | **0.47**    | **Best balance — captures full concepts** |
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
- **Important:** Embedding computation dominates retrieval latency (~2.3 s per query on CPU). FAISS index search itself is sub-millisecond for 33 vectors.

### Index Type: FAISS IndexFlatIP (Exact Search)

With 33 total chunks (10 docs × ~3.3 chunks/doc), exact search is negligibly fast (<1 ms). Approximate indexes (IVF, HNSW) are warranted only at >100k chunks.

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

> **Metric note:** P@3 is computed at the chunk level (each retrieved chunk scored independently). R@3 is computed at the document level using set-based deduplication, so it is bounded [0, 1]. Hit Rate is reported separately.

| Q# | Rank-1 Doc | Top-1 Score | Rank-2 Doc | Rank-3 Doc | GT in Top-3? | P@3    | R@3  |
|----|-----------|-------------|-----------|-----------|-------------|--------|------|
| Q1 | **doc_01** | 0.455 | doc_04 | doc_10 | ✓ | 0.333 | 1.00 |
| Q2 | doc_03 | 0.155 | doc_03 | doc_08 | ✗ | 0.000 | 0.00 |
| Q3 | **doc_03** | 0.527 | doc_07 | doc_06 | ✓ | 0.333 | 1.00 |
| Q4 | doc_01 | 0.459 | **doc_04** | doc_05 | ✓ | 0.333 | 1.00 |
| Q5 | **doc_05** | 0.678 | doc_05 | doc_05 | ✓ | 1.000 | 1.00 |
| Q6 | **doc_06** | 0.574 | doc_06 | doc_03 | ✓ | 0.667 | 1.00 |
| Q7 | **doc_07** | 0.562 | doc_08 | doc_07 | ✓ | 0.667 | 1.00 |
| Q8 | **doc_08** | 0.675 | doc_08 | doc_07 | ✓ | 0.667 | 1.00 |
| Q9 | doc_03 | 0.149 | doc_03 | doc_08 | ✗ | 0.000 | 0.00 |
| Q10 | **doc_10** | 0.574 | doc_10 | doc_10 | ✓ | 1.000 | 1.00 |
| **Avg** | | | | | **8/10** | **0.500** | **0.80** |

> **doc_01** = RAG, **doc_02** = Vector DBs, **doc_03** = LLM Deployment,  
> **doc_04** = Eval Metrics, **doc_05** = Chunking, **doc_06** = Fine-tuning,  
> **doc_07** = MLOps, **doc_08** = Monitoring, **doc_09** = Embeddings, **doc_10** = Prompting

### 2.3 Metric Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Hit Rate@3 (doc-level) | **0.80** (8/10) | Q2, Q9 failed to retrieve ground truth doc |
| Hit Rate@1 (doc-level) | **0.70** (7/10) | Q2, Q4, Q9 missed rank-1; Q2 and Q9 full misses |
| Precision@3 (chunk-level) | **0.500** | 1.0 for focused queries (Q5, Q10); 0.0 for retrieval failures |
| Recall@3 (doc-level) | **0.80** | Set-based deduplication; bounded [0, 1] |

---

## 3. Qualitative Grounding Analysis

### 3.1 Well-Grounded Responses (Q1, Q3, Q5, Q6, Q7, Q8, Q10)

**Q1 — RAG and hallucination reduction**
- doc_01 retrieved at rank 1 (score 0.455). Answer correctly stated RAG reduces hallucination because responses are grounded in retrieved evidence.
- **Verdict: Fully grounded.**

**Q3 — Local LLM deployment**
- doc_03 at rank 1 (score 0.527). Answer correctly cited Ollama and the `ollama pull mistral:7b-instruct` command.
- **Verdict: Fully grounded.**

**Q5 — Chunking strategy**
- All 3 retrieved chunks from doc_05 (score 0.678). Answer described fixed-size character chunking with 512-char chunks and 100-char overlap.
- A minor generalisation ("provides a good balance") slightly overstates the document's comparative claim.
- **Verdict: Mostly grounded — minor overstatement.**

**Q6 — LoRA parameter reduction**
- Two chunks from doc_06, one from doc_03. Answer correctly stated ΔW = AB with 99% parameter reduction.
- **Verdict: Fully grounded.**

**Q7 — Reproducible ML experiments**
- doc_07 retrieved at rank 1 (score 0.562) and rank 3. Answer comprehensively listed experiment tracking (MLflow, W&B, DVC), model versioning, CI/CD, feature stores, and container deployment — all from doc_07.
- **Verdict: Fully grounded.** Generation was slow (86.5 s) due to long output.

**Q8 — Data drift detection**
- doc_08 retrieved at rank 1 (score 0.675) and rank 2. Strongest retrieval result.
- Answer accurately listed PSI, Kolmogorov-Smirnov, chi-squared, MMD, Evidently AI, WhyLogs — all from doc_08.
- **Verdict: Fully grounded.**

**Q10 — Chain-of-thought prompting**
- All 3 chunks from doc_10 (score 0.574). Answer accurately described CoT trigger phrases, effective model sizes (>10B), and self-consistency sampling.
- Generator added ReAct and Tree-of-Thoughts which are not in the retrieved context.
- **Verdict: Mostly grounded — minor parametric bleed-through.**

### 3.2 Partially Grounded Responses (Q4)

**Q4 — RAG evaluation metrics**
- doc_01 ranked above doc_04 (scores ~0.459 vs slightly lower). Both retrieved in top-3.
- Answer correctly listed Precision@k, Recall@k, MRR, nDCG — sourced from doc_04 at rank 2.
- **Verdict: Grounded despite rank-1 misorder.** Close similarity scores caused the rank flip; answer quality was not degraded.

### 3.3 Retrieval Failures (Q2, Q9)

**Q2 — FAISS vs. Chroma**
- doc_02 (ground truth) was **not retrieved** in the top-3. Retrieved docs were doc_03 (LLM Deployment, twice) and doc_08 (Monitoring), all with very low scores (top-1: 0.155).
- **Root cause:** The query "differences between FAISS and Chroma" unexpectedly matched deployment and monitoring vocabulary more strongly than the vector database document. This may reflect embedding non-determinism or vocabulary overlap between "similarity search" and "inference serving" terminology.
- The model correctly abstained: *"The context provided does not contain information about FAISS or Chroma."*
- **Verdict: Retrieval failure. Model correctly abstained — no hallucination.**

**Q9 — Sentence embeddings and semantic similarity**
- doc_09 (ground truth) was **not retrieved** in the top-3. Retrieved docs were doc_03 (LLM Deployment, twice) and doc_08 (Monitoring), top-1 score 0.149.
- **Root cause:** Generic ML vocabulary in the query ("embeddings", "semantic") matched deployment and monitoring documents. doc_09 was not discriminable enough given the small 33-chunk corpus.
- The model gave a partially relevant answer by picking up on the word "embedding similarity" in doc_08's monitoring section, but this is tangential.
- **Verdict: Retrieval failure. Partial parametric drift in answer.**

### 3.4 Hallucination Summary

| Query | Hallucination? | Type | Attribution |
|-------|---------------|------|-------------|
| Q1 | None | — | Fully grounded |
| Q2 | None | Model abstained | Retrieval failure — GT doc not retrieved |
| Q3 | None | — | Fully grounded |
| Q4 | None | — | Grounded via rank-2 doc |
| Q5 | Minor | Slight overstatement | Generation drift |
| Q6 | None | — | Fully grounded |
| Q7 | None | — | Fully grounded |
| Q8 | None | — | Fully grounded (strongest retrieval) |
| Q9 | Minor | Tangential answer | Retrieval failure — off-topic context |
| Q10 | Moderate | Parametric bleed | ReAct/ToT not in retrieved context |

---

## 4. Error Attribution

| Failure Type | Count | Queries Affected |
|-------------|-------|-----------------|
| Retrieval failure (GT doc not in top-3) | 2 | Q2, Q9 |
| Rank-1 miss (GT retrieved but not top-ranked) | 1 | Q4 |
| Generation drift (minor overstatement) | 1 | Q5 |
| Generation drift (moderate — parametric bleed) | 1 | Q10 |
| Clean end-to-end | 5 | Q1, Q3, Q6, Q7, Q8 |

**Key finding:** Two queries (Q2, Q9) suffered complete retrieval failures — the ground truth document did not appear in the top-3 at all, both with top-1 scores below 0.16. One query (Q4) had a rank-1 miss where doc_01 scored marginally above doc_04 (0.459 vs slightly lower), but both were retrieved and the correct answer was generated. Generation errors (Q5, Q10) were minor.

**Q2 retrieval failure analysis:** The query vocabulary ("FAISS", "Chroma", "vector search") strongly overlaps with deployment terminology in doc_03, causing a false positive. A hybrid BM25+dense approach would likely recover doc_02 via keyword matching on "FAISS" and "Chroma".

**Q9 retrieval failure analysis:** Generic embedding vocabulary appeared across multiple documents, making doc_09 non-discriminable. Adding per-document metadata tags or semantic chunking would improve discriminability.

**Note:** Q8 (data drift detection) was the strongest retrieval — doc_08 ranked 1st and 2nd with scores 0.675 and 0.531.

---

## 5. Latency Measurements

All measurements taken on Apple M2 Pro, 16 GB RAM. Ollama using Metal (GPU acceleration).

### 5.1 Per-Stage Latency (averaged over 10 queries)

| Stage | Min | Mean | Max |
|-------|-----|------|-----|
| Query embedding + FAISS search | 1,273 ms | 2,287 ms | 4,812 ms |
| LLM generation | 9,100 ms | 23,070 ms | 86,500 ms |
| **End-to-end** | **11,100 ms** | **25,400 ms** | **88,000 ms** |

> **Note on retrieval latency:** The ~2.3 s retrieval time is dominated by CPU-based sentence embedding computation, not FAISS search. FAISS over 33 vectors is sub-millisecond. On a CUDA GPU, embedding latency would drop to ~5–10 ms.

### 5.2 Per-Query End-to-End Latency

| Q# | Query Type | Retrieval (ms) | Generation (s) | End-to-End (s) |
|----|-----------|---------------|----------------|----------------|
| Q1 | Factual | 3,623 | 11.3 | 14.9 |
| Q2 | Comparative | 1,525 | 27.1 | 28.6 |
| Q3 | Procedural | 4,812 | 14.5 | 19.3 |
| Q4 | Factual | 1,970 | 9.1 | 11.1 |
| Q5 | Design | 1,686 | 17.0 | 18.7 |
| Q6 | Technical | 1,493 | 23.3 | 24.8 |
| Q7 | Best-practice | 1,469 | 86.5 | 88.0 |
| Q8 | Procedural | 3,561 | 9.7 | 13.3 |
| Q9 | Conceptual | 1,273 | 22.0 | 23.3 |
| Q10 | Conceptual | 1,454 | 10.2 | 11.7 |
| **Mean** | | **2,287** | **23.1** | **25.4** |

### 5.3 Model Setup Reference

| Parameter | Value |
|-----------|-------|
| Model name | mistral:7b-instruct |
| Model size | 7B parameters (4-bit quantised via Ollama) |
| Serving stack | Ollama with Metal backend |
| Hardware | Apple M2 Pro, 16 GB unified memory |
| Typical generation latency | 9–87 s per response |

### 5.4 Latency Observations

- **Retrieval varies (1.3–4.8 s)** — driven by CPU embedding computation; early queries (Q1, Q3, Q8) had higher retrieval latency due to model warm-up.
- **Q7 generation was slowest (86.5 s)** — the MLOps best-practices query produced a very long structured list response (~15 tokens/s × ~1,300 tokens). A `max_tokens=512` cap would prevent this.
- **Generation dominates (>90% of wall time)** — 7B quantised model on Apple Silicon.
- **Variance in generation (~9.5×)** is driven by response length, not input complexity.
- For production use, GPU serving (vLLM on A10G) would reduce generation to ~1–2 s and GPU-based embedding would drop retrieval to <10 ms.

---

## 6. Summary and Recommendations

| Aspect | Value | Notes |
|--------|-------|-------|
| Hit Rate@3 (doc-level) | **0.80** | 8/10 queries retrieved correct doc in top-3 |
| Hit Rate@1 (doc-level) | **0.70** | Q2, Q4, Q9 missed rank-1; Q2 and Q9 full misses |
| Precision@3 (chunk-level avg) | **0.500** | 1.0 for focused queries; 0 for failures |
| Recall@3 (doc-level avg) | **0.80** | Set-based deduplication; bounded [0, 1] |
| Generation faithfulness | 5/10 clean | 2 retrieval failures, 1 rank miss, 2 minor generation drifts |
| Mean retrieval latency | **2,287 ms** | CPU embedding bottleneck; GPU would reduce to <10 ms |
| Mean generation latency | **23.1 s** | Outlier Q7 (86.5 s); median ~17 s |
| Mean end-to-end | **25.4 s** | Dominated by generation |

**Top-priority improvements:**
1. **Add hybrid search (BM25 + dense)** — Q2 and Q9 both failed because dense-only retrieval couldn't discriminate on specific entity names (FAISS, Chroma) or conceptual vocabulary. Keyword matching would recover these.
2. **Cap generation length** — Add `max_tokens=512` to prevent Q7-style 86 s responses.
3. **Add document metadata tags** — Tagging chunks with topic labels (e.g., `#vector-db`, `#embeddings`) enables metadata-filtered retrieval for ambiguous queries.
4. **Use GPU embedding** — Reduce retrieval from ~2.3 s to <10 ms per query.
5. **Add low-confidence fallback** — Queries with top-1 score < 0.20 (Q2: 0.155, Q9: 0.149) should trigger a fallback or report low-confidence to the user.
