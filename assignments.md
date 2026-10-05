# Assignments

Every student does **two parts**:

- **Part A: Search Evaluation.** The same for everyone.
- **Part B: Your RAG topic.** Different for every student. Find your name in the [table below](#part-b-your-rag-topic).

Part B builds on Part A: you use the metrics from Part A to prove whether your Part B technique made the search better or worse.

## Part A: Search Evaluation (everyone)

Right now the project can search, but it cannot tell us **how good** the search is. Your job is to measure it.

### Step 1: Build a small test set

Create `data/eval_set.json` with **at least 15 questions** about the fitness PDFs. For each question, list the chunks that are a correct answer, with a relevance grade:

```json
[
  {
    "question": "how much protein should i eat to build muscle?",
    "relevant": {"<chunk-id-1>": 2, "<chunk-id-2>": 1}
  }
]
```

- Grade `2` = fully answers the question, `1` = partly relevant. Chunks you do not list count as `0`.
- For Precision, Recall, F1, MRR and MAP, any grade above 0 counts as "relevant". NDCG uses the grades themselves.
- Mix easy questions, hard questions, and a few with more than one relevant chunk.
- Hint: `retrieve()` in `rag/retriever.py` does not return chunk ids yet. ChromaDB does return them in `results["ids"]`, so add an `id` field to `RetrievedChunk`.

### Step 2: Implement the metrics

Create `src/fitness_agent/evaluation/metrics.py`. Write every metric **yourself in plain Python**; do not import them from a library (scikit-learn, ranx, RAGAS and so on). Each metric is computed on the top-K results.

| Metric | What it tells you | Formula |
| --- | --- | --- |
| Precision@K | How much of what we returned is relevant | relevant results in top K / K |
| Recall@K | How much of the relevant material we found | relevant results in top K / total relevant chunks |
| F1@K | One number balancing the two | 2 x P x R / (P + R) |
| MRR | How early the first relevant result appears | average over questions of 1 / rank of first relevant result (0 if none) |
| MAP | How well all relevant results are ranked | average over questions of AP, where AP = sum of Precision@i at every rank i that holds a relevant result / total relevant chunks |
| NDCG@K | Ranking quality with graded relevance | DCG@K / IDCG@K, where DCG@K = sum over ranks i of grade_i / log2(i + 1), and IDCG@K is the DCG of the perfect ordering |

Handle the edge cases: no relevant chunks retrieved, no results at all, and P + R = 0.

### Step 3: Test the metrics

Add `tests/test_evaluation.py`. For each metric, work out a small example by hand and check that your function returns the same number. The tests must pass with `pytest` and must not need an LLM or the database.

### Step 4: Write the evaluation script

Create `scripts/05_evaluate.py`, in the same style as the other lesson scripts. It should:

1. Load `data/eval_set.json`.
2. Run the search pipeline for every question.
3. Print a table with all six metrics for `K = 1, 3, 5, 10`.
4. Print the three questions with the worst scores, so you can see where the search fails.

### Step 5: Answer these questions in your report

1. Which metric is the most useful for this project, and why?
2. Run the evaluation with and without the query reformation step. Did it help?
3. When Precision goes up as K changes, what happens to Recall? Show it with your numbers.
4. Pick one of your worst questions. Why did the search fail on it?

## Part B: Your RAG topic

| # | Student | Topic |
| --- | --- | --- |
| 1 | Nidhi Mittal | Chunking strategies |
| 2 | Jitendra Kumar Saroj | Semantic chunking |
| 3 | Vivek Harle | Embedding model comparison |
| 4 | Aditya Venkata Satyanarayana Mokkapati | Hybrid search (BM25 + vector) |
| 5 | Radharapu Bharath Kumar | Multi-query retrieval and RAG-Fusion |
| 6 | Avinash Dupaguntla | Reranking with a cross-encoder |
| 7 | Kailas Kanade | HyDE (Hypothetical Document Embeddings) |
| 8 | Raviraj Deshpande | Query decomposition and step-back prompting |
| 9 | Sheetal Deshpande | Metadata filtering and self-querying |
| 10 | Vishal Kailas Kharade | Parent-document (small-to-big) retrieval |
| 11 | Usama Mirkar | Sentence-window retrieval |
| 12 | Hariharan | Contextual compression |
| 13 | Harmeet Bedi | Maximal Marginal Relevance (MMR) |
| 14 | Valathappan Sivaraman | Contextual retrieval |
| 15 | Balaji Kumar | Corrective RAG (CRAG) |
| 16 | Surendran Sundarababu | Adaptive RAG (query routing) |
| 17 | Mohit Luthra | Answer evaluation: faithfulness and relevancy |
| 18 | Shirish Suryakant Pathak | Citations and source attribution |
| 19 | Bhanupriya | Conversational RAG |
| 20 | Jolly Shringi | Semantic caching |

### What each topic asks for

1. **Chunking strategies.** Add sentence-based and recursive chunking next to the current fixed-size word chunking in `rag/chunker.py`. Compare all three at two or three chunk sizes.
2. **Semantic chunking.** Split the text where the meaning changes: embed the sentences and start a new chunk when the similarity between neighbours drops. Compare with the current chunker.
3. **Embedding model comparison.** Ingest the PDFs with at least three embedding models (the ChromaDB default plus two others, such as `bge-small` or `nomic-embed-text` through Ollama). Compare quality, speed and index size.
4. **Hybrid search (BM25 + vector).** Add keyword search with BM25 and combine it with the vector search using a weighted score. Find questions where keyword search wins and where vector search wins.
5. **Multi-query retrieval and RAG-Fusion.** Have the LLM write several versions of the question, search with each, and merge the result lists with Reciprocal Rank Fusion.
6. **Reranking with a cross-encoder.** Retrieve a larger set (for example top 20), rerank it with a cross-encoder model, and keep the best K. Measure the extra latency.
7. **HyDE.** Have the LLM write a made-up answer to the question, then search with the embedding of that answer instead of the question.
8. **Query decomposition and step-back prompting.** Break a complex question into simpler sub-questions, retrieve for each, and combine. Also try step-back prompting: ask a more general question first. Add a few multi-part questions to your test set.
9. **Metadata filtering and self-querying.** Store more metadata at ingestion (for example a topic or document type). Have the LLM turn the question into a search query plus a metadata filter, and pass that filter to ChromaDB.
10. **Parent-document retrieval.** Search over small chunks, but send the larger parent chunk (or the full page) to the LLM. Compare with searching the large chunks directly.
11. **Sentence-window retrieval.** Embed single sentences, and when one matches, send it to the LLM together with the sentences around it. Try different window sizes.
12. **Contextual compression.** After retrieval, keep only the sentences of each chunk that are relevant to the question before building the context. Measure how many tokens you save and whether answers stay correct.
13. **Maximal Marginal Relevance (MMR).** Implement MMR yourself to pick results that are relevant but not repeats of each other. Show how the results change as you move the lambda setting.
14. **Contextual retrieval.** Before embedding each chunk, have the LLM write one or two sentences that place the chunk in its document, and prepend them to the chunk. Re-ingest and compare.
15. **Corrective RAG (CRAG).** Add a LangGraph node that grades the retrieved chunks. If they are poor, rewrite the query and retrieve again (with a retry limit) before generating.
16. **Adaptive RAG (query routing).** Add a router node that decides per question: answer without retrieval, do one retrieval, or do a multi-step retrieval. Show the route taken for different kinds of questions.
17. **Answer evaluation: faithfulness and relevancy.** Part A measures the search; you measure the answer. Build an LLM-judge evaluation that scores faithfulness (is every claim supported by the context?) and answer relevancy (does it answer the question?).
18. **Citations and source attribution.** Make the generator cite the chunk behind every claim, like `[1]`, and write a check that each citation really supports its sentence. Report how often citations are correct.
19. **Conversational RAG.** Make the agent handle follow-up questions ("and how much for women?") by rewriting them into standalone questions using the chat history. Build a small test set of multi-turn conversations.
20. **Semantic caching.** Cache answers by question embedding, and reuse an answer when a new question is similar enough. Measure hit rate and time saved, and find the similarity threshold where wrong answers start being served.

### What to deliver for Part B

1. **Working code** in this project, following the existing structure: new logic in `src/fitness_agent/`, a runnable script in `scripts/`, and at least two unit tests.
2. **A before-and-after comparison** using your Part A metrics: the baseline pipeline against the pipeline with your technique, on the same test set. Topics 17 to 20 are not only about search ranking, so use the measure named in the topic as well.
3. **A short explanation** in your report: what the technique is, what problem it solves, when it helps, and when it hurts (cost, latency, complexity).

A result that shows your technique did **not** help is a perfectly good result, as long as you measured it properly and can explain why.

## How to submit

1. Fork this repository and work on a branch named `assignment/<your-name>`.
2. Put your report in `reports/<your-name>.md`. It should hold your Part A metrics table and answers, and your Part B comparison and explanation.
3. Make sure `pytest` passes.
4. Share the link to your fork with the instructor.
5. Be ready to give a 5-minute demo of your topic to the class, so that everyone learns all 20 techniques.

## How you will be graded

| Area | Weight |
| --- | --- |
| Part A: metrics are correct and tested | 30% |
| Part A: test set quality and analysis | 15% |
| Part B: working implementation | 30% |
| Part B: before-and-after evaluation | 15% |
| Report clarity and demo | 10% |
