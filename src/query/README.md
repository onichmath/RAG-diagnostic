# Query Transformation Stack

This module adds query-side transformations that run before retrieval. It uses `google.colab.ai` models to improve recall and ranking without rebuilding the FAISS index.

## What it does
- **Rewrite**: Normalize the user question into a concise, unambiguous clinical query.
- **Expansion**: Suggest alternate phrasings/synonyms/related terms to cast a wider net.
- **Decomposition**: Break complex questions into atomic sub-questions for targeted retrieval.
- **Step-back**: Generate a higher-level variant to pull guideline-style context.

`QueryTransformer` asks a single Colab AI model call to return all of the above as JSON, parses it, and builds a deduplicated list of candidate queries. Retrieval then issues each variant in order and merges results (deduped by `doc_id`) until `k` results are collected.

## How it is used in evaluation
`src/eval/eval.py` now:
1. Instantiates `QueryTransformer()` once.
2. Transforms every test query via `transformer.transform(query_text)`.
3. Runs `multi_query_similarity_search` with the transformed variants before rerankers execute.

The rest of the pipeline (indexing, reranking, metrics) is unchanged.

## Colab AI dependency and fallbacks
- The code attempts `from google.colab import ai` with the default model `google/gemini-2.5-flash`.
- If Colab AI is unavailable, the transformer falls back to pass-through behavior so evaluation still runs.

## Extending or tuning
- Adjust `max_expansions`, `max_subqueries`, or `model_name` when constructing `QueryTransformer`.
- Modify `_build_prompt` in `src/query/query_transformer.py` to change formatting or add domain-specific hints.
