# Assignment: Your Own RAG Use Case, With Evaluation

This project is a **fitness** question-answering agent. In this assignment you turn it into an agent for a **different use case**, and then measure how good its search is.

- **The use case is different for every student.** Find your name in the [table below](#your-use-case).
- **The project stays as it is.** Keep the same pipeline: ingestion, search, the three guardrails, and the LangGraph agent. You are not asked to add new RAG techniques. You change the documents, the prompts and the guardrail data so the agent works for your use case.
- **Evaluation is part of everyone's project.** Everyone implements the search metrics we learned: Precision, Recall, F1 Score, MRR, MAP and NDCG.

## Your use case

| # | Student | Use case | The agent answers questions about |
| --- | --- | --- | --- |
| 1 | Nidhi Mittal | HR policy assistant | Leave, benefits and conduct rules from an employee handbook |
| 2 | Jitendra Kumar Saroj | Banking product assistant | Savings accounts, cards and loans: fees, eligibility and terms |
| 3 | Vivek Harle | Health insurance assistant | What a policy covers, exclusions, waiting periods and claims |
| 4 | Aditya Venkata Satyanarayana Mokkapati | Consumer rights assistant | Consumer protection law: rights, complaints and refunds |
| 5 | Radharapu Bharath Kumar | Income tax helper | Deductions, filing steps and deadlines from official tax guides |
| 6 | Avinash Dupaguntla | Diabetes patient education assistant | Diet, monitoring and lifestyle from public health leaflets |
| 7 | Kailas Kanade | Medicine information assistant | Usage, side effects and storage from patient information leaflets |
| 8 | Raviraj Deshpande | Visa and passport assistant | Application steps, documents and fees |
| 9 | Sheetal Deshpande | University admissions assistant | Courses, eligibility, fees and dates from a prospectus |
| 10 | Vishal Kailas Kharade | Car owner's manual assistant | Features, warning lights and maintenance from a vehicle manual |
| 11 | Usama Mirkar | IT helpdesk assistant | Password, VPN, laptop and software how-to guides |
| 12 | Hariharan | Software documentation assistant | How to use one tool or library (for example Git or Docker) from its docs |
| 13 | Harmeet Bedi | Cooking and recipe assistant | Ingredients, steps, substitutions and timings from cookbooks |
| 14 | Valathappan Sivaraman | Farming advisory assistant | Crop care, sowing seasons, pests and fertiliser from farming guides |
| 15 | Balaji Kumar | Driving rules assistant | Traffic rules, road signs and licence steps from a driver's handbook |
| 16 | Surendran Sundarababu | Research paper assistant | Methods and findings from a small set of research papers |
| 17 | Mohit Luthra | Annual report analyst | Revenue, risks and strategy from company annual reports |
| 18 | Shirish Suryakant Pathak | Home rental and tenancy assistant | Rent agreements, deposits and tenant and owner rights |
| 19 | Bhanupriya | E-commerce support assistant | Shipping, returns, refunds and warranty policies |
| 20 | Jolly Shringi | Government schemes assistant | Eligibility, benefits and how to apply for public schemes |

You choose the actual documents. Use **3 to 5 public PDFs** that contain real text (not scanned images) and no private or confidential information.

## Step 1: Move the project to your use case

1. **Documents.** Replace the PDFs in `data/pdfs/` with your own and run `python scripts/01_ingest.py`.
2. **Collection.** Change `collection_name` in `config.py`, so your chunks do not mix with the fitness chunks.
3. **Prompts.** Rewrite the prompts so they describe your assistant, not a fitness assistant:
   - `rag/query_rewriter.py` (`REWRITE_PROMPT`)
   - `rag/generator.py` (`ANSWER_PROMPT`)
   - `guardrails/output_guard.py` (`JUDGE_PROMPT`)
4. **NLU guardrail.** Rewrite `data/guardrail_training.csv` for your use case:
   - Replace the `fitness` rows with at least 60 in-domain questions under your own label, and set `ALLOWED_LABEL` in `guardrails/nlu_guard.py` to match.
   - Update the `off_topic` and `harmful` rows. What counts as harmful depends on your use case: for a medicine assistant it may be overdose questions, for a banking assistant it may be fraud.
5. **Regex guardrail.** Check the patterns in `guardrails/regex_guard.py`. Does your use case need a new one (for example an account number or a policy number)? Add at least one pattern that makes sense for your domain.
6. **Refusal messages.** Update the messages in `agent/nodes.py`.
7. **Run all four lessons** (`01_ingest.py` to `04_agent.py`) and check each one works on your use case. Update the tests where needed so `pytest` still passes.

## Step 2: Evaluate the search

Right now the project can search, but it cannot tell us **how good** the search is. You will measure it.

### 2.1 Build a test set

Create `data/eval_set.json` with **at least 15 questions** about your documents. For each question, list the chunks that are a correct answer, with a relevance grade:

```json
[
  {
    "question": "<a question a real user would ask>",
    "relevant": {"<chunk-id-1>": 2, "<chunk-id-2>": 1}
  }
]
```

- Grade `2` = fully answers the question, `1` = partly relevant. Chunks you do not list count as `0`.
- For Precision, Recall, F1, MRR and MAP, any grade above 0 counts as "relevant". NDCG uses the grades themselves.
- Mix easy questions, hard questions, and a few with more than one relevant chunk.
- Chunk ids look like `<file name>-p<page>-c<index>` (see `rag/chunker.py`). `retrieve()` in `rag/retriever.py` does not return them yet, but ChromaDB does in `results["ids"]`, so add an `id` field to `RetrievedChunk`.
- If you change `CHUNK_SIZE` or `CHUNK_OVERLAP`, the chunk ids change and you must label again.

### 2.2 Implement the metrics

Create `src/fitness_agent/evaluation/metrics.py`. Write every metric **yourself in plain Python**; do not import them from a library. Each metric is computed on the top-K results.

| Metric | What it tells you | Formula |
| --- | --- | --- |
| Precision@K | How much of what we returned is relevant | relevant results in top K / K |
| Recall@K | How much of the relevant material we found | relevant results in top K / total relevant chunks |
| F1@K | One number balancing the two | 2 x P x R / (P + R) |
| MRR | How early the first relevant result appears | average over questions of 1 / rank of first relevant result (0 if none) |
| MAP | How well all relevant results are ranked | average over questions of AP, where AP = sum of Precision@i at every rank i that holds a relevant result / total relevant chunks |
| NDCG@K | Ranking quality with graded relevance | DCG@K / IDCG@K, where DCG@K = sum over ranks i of grade_i / log2(i + 1), and IDCG@K is the DCG of the perfect ordering |

Handle the edge cases: no relevant chunks retrieved, no results at all, and P + R = 0.

### 2.3 Test the metrics

Add `tests/test_evaluation.py`. For each metric, work out a small example by hand and check that your function returns the same number. The tests must not need an LLM or the database.

### 2.4 Write the evaluation script

Create `scripts/05_evaluate.py`, in the same style as the other lesson scripts. It should:

1. Load `data/eval_set.json`.
2. Run the search pipeline for every question.
3. Print a table with all six metrics for `K = 1, 3, 5, 10`.
4. Print the three questions with the worst scores, so you can see where the search fails.

## Step 3: Write your report

Create `reports/<your-name>.md` with:

1. **Your use case.** Who the users are, which documents you chose and why.
2. **What you changed** to move the project to your use case, including your guardrail decisions: what is in-domain, what is off topic, and what is harmful.
3. **Your metrics table** from `05_evaluate.py`.
4. **Answers to these questions:**
   1. Which metric matters most for your use case, and why? (Is it worse for your users to miss a relevant chunk, or to see an irrelevant one?)
   2. Run the evaluation with and without the query reformation step. Did it help?
   3. As K grows, what happens to Precision and Recall? Show it with your numbers.
   4. Pick one of your worst questions. Why did the search fail on it?
   5. Show one input each guardrail blocked correctly, and one mistake a guardrail made.

## How to submit

1. Fork this repository and work on a branch named `assignment/<your-name>`.
2. Make sure `pytest` passes and all five scripts run.
3. Share the link to your fork with the instructor.
4. Be ready to give a 5-minute demo of your agent and your metrics to the class.

## How you will be graded

| Area | Weight |
| --- | --- |
| The agent works end to end on your use case | 25% |
| Guardrails fit your use case | 15% |
| Metrics are correct and tested | 30% |
| Test set quality and analysis in the report | 20% |
| Report clarity and demo | 10% |
