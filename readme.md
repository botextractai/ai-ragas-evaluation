# Retrieval-Augmented Generation Assessment (RAGAS) evaluation of semantic search vs hybrid search

This example implements a comprehensive evaluation system for a Retrieval-Augmented Generation (RAG) pipeline. It demonstrates how developers can leverage evaluation metrics to gain insights and make data-driven decisions to improve their RAG pipelines.

Ragas is an evaluation platform designed specifically for RAG.

This examples integrates ragas to evaluate a RAG pipeline. It generates synthetic ground truth data (data that represents the ideal responses) and establishes a comprehensive set of metrics to evaluate the impact of using hybrid search that combines both keyword search and dense vector semantic search (using the LangChain `EnsembleRetriever`), compared to using just dense vector semantic search alone. It analyses the different stages of RAG, covering retrieval evaluation, generation evaluation, and end-to-end evaluation.

The example RAG pipeline includes a relevance check that determines, if the Large Language Model (LLM) answer is relevant to the question. It only allows answers with a relevance score of 4 or 5 (out of 5 possible, with 5 being the most relevant). If the score is below 4, then the final answer is just "I don't know". This filters out irrelevant answers and also protects from abuse such as malicious prompts (injections), which will generate answers that are irrelevant to the question and potentially harmful.

## Costs

The ragas library uses your LLM Application Programming Interface (API) key (in this example from OpenAI) extensively. The analysis that ragas provides is LLM assisted evaluation, meaning every time a ground truth example is generated or evaluated, an LLM is called (sometimes multiple times for one metric) and an API charge is incurred. If you generate 30 ground truth examples as in this program code, which includes the generation of both questions and answers, and then run six different evaluation metrics, the number of LLM API calls you make multiplies substantially. It is recommended to use it sparingly until you have a good grasp on how often the calls are being made. You can adjust the number of ground truth examples in the program code with the `test_size` setting.

If you want to reduce your costs, then you can use the pre-made Comma-Separated Values (CSV) files from the `backup_data` folder, instead of generating your own data.

## Required API key for this example

You need an OpenAI API key for this example. [Get your OpenAI API key here](https://platform.openai.com/login). Insert the OpenAI API key into the `.env.example` file and then rename this file to just `.env` (remove the ".example" ending).

## The results

This ragas evaluation covers 6 different evaluation metrics (2 metrics each for retrieval, generation, and end-to-end evaluation).

| Stages                    | Metrics              | Descriptions                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Retrieval**             | `context_precision`  | The signal-to-noise ratio of retrieved context. `context_precision` is a metric that evaluates whether all of the ground truth relevant items present in the contexts are ranked higher or not. Ideally, all the relevant chunks must appear at the top ranks. This metric is computed using the question, ground truth, and contexts, with values ranging between 0 and 1, where higher scores indicate better precision.                           |
| **Retrieval**             | `context_recall`     | Can it retrieve all the relevant information required to answer the question? `context_recall` measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance.                                                                         |
| **Generation**            | `faithfullness`      | How factually accurate is the generated answer? This measures the factual consistency of the generated answer against the given context. It is calculated from the answer and retrieved context. The answer is scaled to a (0-1) range, with a higher score being better.                                                                                                                                                                            |
| **Generation**            | `answer_relevancy`   | How relevant is the generated answer to the question? Answer relevancy focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the question, the context, and the answer.                                                                               |
| **End-to-end evaluation** | `answer_correctness` | Gauges the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the ground truth and the answer, with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated answer and the ground truth, signifying better correctness.                                                                                                                                             |
| **End-to-end evaluation** | `answer_similarity`  | Assesses the semantic resemblance between the generated answer and the ground truth. The concept of answer semantic similarity pertains to the assessment of the semantic resemblance between the generated answer and the ground truth. This evaluation is based on the ground truth and the answer, with values falling within the range of 0 to 1. A higher score signifies a better alignment between the generated answer and the ground truth. |

| >>>>> The results will look similar to this example: <<<<< |
| ---------------------------------------------------------- |

```
Performance Comparison:

**Retrieval**:
                      Similarity Run    Hybrid Run    Difference
context_precision           0.752767      0.756282     -0.003515
context_recall              0.941667      0.941667      0.000000

**Generation**:
                      Similarity Run    Hybrid Run    Difference
faithfulness                0.915798      0.944392     -0.028594
answer_relevancy            0.962976      0.960862      0.002114

**End-to-end evaluation**:
                      Similarity Run    Hybrid Run    Difference
answer_correctness          0.731543      0.700431      0.031112
answer_similarity           0.928985      0.929491     -0.000506
```
