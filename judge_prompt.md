# LLM-as-a-Judge Prompt

This is the evaluation prompt used to score every `(question, answer)` pair produced by
the retrieval pipelines. The judge is served with `response_format = json_object` at
temperature 0. `{question}` and `{answer}` are substituted at runtime.

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
