# A Comparative Evaluation of Retrieval Pipelines for Large-Scale Scientific Question Answering with Open-Weight LLMs

This repository accompanies the paper **"A Comparative Evaluation of Retrieval Pipelines for Large-Scale Scientific Question Answering with Open-Weight LLMs"**, submitted to the **IEEE AIxSET** conference.

**Authors:** Bhagyesh Rathi¹, Eshan Chawla¹\*, Aleksander Ershov¹, William B. Andreopoulos¹
¹San José State University, Department of Computer Science · \*Equal Contribution

---

## Overview

Retrieval-Augmented Generation (RAG) is the dominant paradigm for grounding Large Language Models (LLMs) in external knowledge, yet the design space of retrieval pipelines is large and the empirical trade-offs between variants remain poorly characterized — particularly on domain-specific corpora at realistic scale.

This work presents a **controlled comparison of six retrieval strategies** for scientific question answering, all sharing a common stack so that differences in answer quality can be attributed to the retrieval strategy alone:

1. **Classic RAG** (baseline) — top-*k* dense retrieval, then answer directly.
2. **Query Rephrased RAG** — LLM rewrites the query into a clean semantic-search query before retrieval.
3. **Rephrased & Reranked RAG** — query rewriting followed by an LLM cross-encoder–style re-ranking stage.
4. **Fusion RAG (RAG-Fusion / RRF)** — multi-query expansion fused with Reciprocal Rank Fusion.
5. **Tool Call RAG** (agentic) — retrieval exposed as a callable tool; the model decides whether/when to retrieve.
6. **ColBERT RAG** (late-interaction dense retrieval) — token-level MaxSim retrieval over a distractor-heavy pool using ColBERTv2 + PLAID.

All pipelines share the same embedding model, vector store, generator, answer-grounding prompt template, and `top-k = 3` retrieval, and are evaluated with an **LLM-as-a-judge** protocol across multiple quality dimensions.

## Shared Stack

| Component | Choice |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers, GPU) |
| Vector store | ChromaDB (`arxiv_papers` collection), cosine distance, `top-k = 3` |
| Generator (and reranker / agent / fusion backbone) | `meta-llama/Llama-3.1-8B-Instruct`, served with vLLM (OpenAI-compatible, `localhost:8002`), temperature 0, 8192-token context |
| Judge | `Qwen2.5-14B-Instruct-AWQ` (vLLM, `localhost:8003`) |
| ColBERT retriever | ColBERTv2 with the PLAID index (separate GPU) |

Each indexed document is the concatenation of paper title and abstract (`"Title: {title}\nAbstract: {abstract}"`) with submitter, categories, comment, and date retained as metadata.

## Dataset

- **Corpus:** the arXiv corpus released on Zenodo ([record 15808027](https://zenodo.org/records/15808027), ~4.7 GB). Papers dated **2024–2025** are embedded and loaded into Chroma (~463,971 papers).
- **Evaluation set:** a random sample of **10,000 papers from 2025**, drawn via single-pass reservoir sampling (`seed = 42`) over the streamed JSON.
- **Synthetic queries:** for each sampled paper, `Llama-3.1-70B-Instruct` generates **two retrieval-oriented questions** — a **problem query** (the research gap/limitation) and a **method query** (the methodology/technical approach) — yielding ~**20,000 query–gold-paper pairs**. Queries are constrained to 1–2 lines, written in natural language, and prohibited from reusing the paper title verbatim. The source paper is the gold retrieval target.

The released synthetic question dataset is available on Kaggle: https://www.kaggle.com/datasets/bhagyeshrathi/scientific-question-answering-llms

## Evaluation Protocol (LLM-as-a-judge)

For each `(question, answer)` pair the judge first decides `is_answer` — whether the answer makes a genuine attempt (as opposed to refusing or claiming insufficient context). If `is_answer` is false, all metric scores are set to 0; otherwise the judge assigns an integer **1–5** on each of:

- **Accuracy** — factual correctness given the retrieved evidence
- **Completeness** — coverage of the question
- **Faithfulness** — grounding in the retrieved papers (absence of hallucination)
- **Relevance** — on-topic-ness of the answer
- **Clarity** — readability and coherence
- **Overall**

**Judge prompt.** Each `(question, answer)` pair is scored with the following prompt (`response_format = json_object`, temperature 0):

```text
Evaluate the quality of the Answer based on the Question.

First decide whether the Answer actually answers the Question.

Set:
- "is_answer": true if the Answer makes a real attempt to answer the Question.
- "is_answer": false if the Answer says there is not enough context, refuses to answer, is empty, irrelevant, or does not provide an actual answer.

Scoring rules:
- If "is_answer" is false, set all score fields to 0.
- If "is_answer" is true, score each metric from 1 to 5.

Question:
{question}

Answer:
{answer}

Provide your evaluation strictly in the following JSON format.
Do not add markdown, explanation, or extra text.

{
    "is_answer": true,
    "accuracy_score": 1,
    "completeness_score": 1,
    "faithfulness_score": 1,
    "relevance_score": 1,
    "clarity_score": 1,
    "overall_score": 1
}
```

**Aggregation.** Results are reported separately for the **problem** and **method** query subsets. We report (i) the **answer rate** — the percentage of queries with `is_answer = true` — and (ii) the mean of each 1–5 metric with **zero-valued scores excluded**, so per-metric averages reflect quality *conditional on the model attempting an answer*, while the answer rate captures how often it attempts one. Each strategy is evaluated on ≈9,900–10,050 papers (≈19,800–20,100 queries).

## Key Results

**Overall judge score (1–5, zeros excluded):**

| Strategy | Problem overall | Method overall | Mean overall |
|---|---|---|---|
| Classic RAG | 4.85 | 4.83 | 4.84 |
| Query Rephrased RAG | 4.62 | 4.76 | 4.69 |
| **Rephrased & Reranked RAG** | **4.89** | **4.95** | **4.92** |
| Fusion RAG | 4.78 | 4.87 | 4.82 |
| Tool Call RAG | 4.84 | 4.92 | 4.88 |
| ColBERT RAG | 2.66 | 2.38 | 2.52 |

**Summary of findings:**

1. **Query rewriting + LLM re-ranking** is the best pipeline (overall 4.92/5), beating the classic baseline on every metric for both query types.
2. **Agentic Tool Call RAG** is competitive (4.88) and reaches a 100% answer rate.
3. The **classic single-shot baseline is already strong** (4.84); more elaborate Chroma strategies add only modest gains.
4. **Query rewriting without re-ranking** underperforms (4.69) — the re-ranking stage is what makes rewriting pay off.
5. A **stronger retriever did not yield better answers here**: ColBERT scored lowest (2.52/5) and answered fewer queries, most plausibly because its 60K distractor-heavy pool caused top-3 retrieval to surface off-target papers. This (retriever pool composition, indexing, and comparison fairness against the Chroma collection) is the result most worth interrogating.

## Repository Contents

| File | Description |
|---|---|
| `Personalization_Metric_final.ipynb` | Final, fully reproducible notebook: vector store setup, synthetic query generation, all six retrieval pipelines, LLM-as-a-judge evaluation, and visualization. |
| `Personalization_Metric.ipynb` | Earlier development notebook (retained for history). |
| `personalization metrics.pdf` | The paper manuscript. |

## Reproducing the Experiments

The notebook (`Personalization_Metric_final.ipynb`) is organized to run top-to-bottom and is structured as follows:

1. **GPU / environment check** and dependency installation (`torch`, `chromadb`, `sentence-transformers`, `vllm`, `openai`, `ijson`, `colbert-ai`, …).
2. **Vector store setup** — download the Zenodo arXiv corpus, filter to 2024–2025, embed with `all-MiniLM-L6-v2`, and persist to ChromaDB.
3. **Model serving** — launch `Llama-3.1-8B-Instruct` (generator, `:8002`) and `Qwen2.5-14B-Instruct-AWQ` (judge, `:8003`) with vLLM.
4. **Synthetic dataset generation** — reservoir-sample 10K papers (2025, seed 42) and generate problem/method queries with `Llama-3.1-70B-Instruct`.
5. **Retrieval pipelines** — Classic, Query Rephrased, Rephrased & Reranked, RAG Fusion (RRF), Tool Call (agentic), and ColBERT. Each writes its answers to `arxiv_2025_llama_8b_I_<strategy>_rag.jsonl`.
6. **LLM-as-a-judge evaluation** — scores each `*_rag.jsonl` into a parallel `*_rag_eval.jsonl` (resumable).
7. **Visualization** — per-metric and per-strategy answer-rate / overall-quality charts.

**Requirements:** CUDA-capable GPU(s) (the experiments were run on RTX 4090–class hardware), and sufficient disk for the ~4.7 GB corpus plus the Chroma index and ColBERT PLAID index.

> **Note:** because the generator and judge are served as local vLLM OpenAI-compatible endpoints (`localhost:8002` / `localhost:8003`), start those servers before running the generation and evaluation sections.

## Limitations

As discussed in the paper:

1. The evaluation relies on a judge model that shares a backbone family with the generator, potentially introducing self-preference bias.
2. Gold relevance is defined by synthetic queries generated from each paper's abstract, which may capture a simplified view of research relevance.
3. Metric aggregation excludes zero-valued scores, requiring a joint reading of answer rates and conditional quality scores.
4. ColBERT was evaluated against a different document pool and index structure than the Chroma-based pipelines, which may confound the comparison.
5. All pipelines use a single 8B-parameter generator at temperature 0, which may limit generalizability to larger or more creative generation settings.

## Citation

```bibtex
@inproceedings{rathi2026retrieval,
  title     = {A Comparative Evaluation of Retrieval Pipelines for Large-Scale Scientific Question Answering with Open-Weight LLMs},
  author    = {Rathi, Bhagyesh and Chawla, Eshan and Ershov, Aleksander and Andreopoulos, William B.},
  booktitle = {Proceedings of the IEEE AIxSET Conference},
  year      = {2026}
}
```

## Code and Data Availability

- **Code:** https://github.com/wandreopoulos/personalized_evaluation_metrics
- **Dataset:** https://www.kaggle.com/datasets/bhagyeshrathi/scientific-question-answering-llms
