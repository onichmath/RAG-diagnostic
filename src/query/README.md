# Query Transformation Stack

This module adds query-side transformations that run before retrieval. It tries `google.colab.ai` first, then falls back to Gemini (`google.genai`) if `google.colab.ai` is unavailable, to improve recall/ranking without rebuilding the FAISS index.

## What it does
- **Rewrite**: Normalize the user question into a concise, unambiguous clinical query.
- **Expansion**: Suggest alternate phrasings/synonyms/related terms to cast a wider net.
- **Decomposition**: Break complex questions into atomic sub-questions for targeted retrieval.
- **Step-back**: Generate a higher-level variant to pull guideline-style context.

`QueryTransformer` asks a single model call to return all of the above as JSON (Colab AI first, Gemini fallback), parses it, and builds a deduplicated list of candidate queries. Retrieval then issues each variant in order and merges results (deduped by `doc_id`) until `k` results are collected.

## How it is used in evaluation
`src/eval/eval.py` now:
1. Instantiates `QueryTransformer()` once.
2. Transforms every test query via `transformer.transform(query_text)`.
3. Runs `multi_query_similarity_search` with the transformed variants before rerankers execute.

The rest of the pipeline (indexing, reranking, metrics) is unchanged.

## Runtime dependency and fallbacks
- First tries `google.colab.ai` (`gemini-2.5-flash` by default, automatically prefixed with `google/` for Colab AI).
- If Colab AI is unavailable, falls back to the Gemini API via `google.genai`, which requires `GEMINI_API_KEY` in the environment.
- If neither is available or calls fail, the transformer becomes a no-op so evaluation still runs.

**Note on model names**: The `model_name` parameter accepts the base model name (e.g., `"gemini-2.5-flash"`). The code automatically handles the different formats required by each API:
- Colab AI uses `"google/gemini-2.5-flash"` format
- Gemini API uses `"gemini-2.5-flash"` format

### Setting `GEMINI_API_KEY` in Colab (avoid storing in repo)
Run once per session before `build_pipeline.py`:
```python
import os
# Option 1: paste manually (not persisted in files)
os.environ["GEMINI_API_KEY"] = "your-key-here"

# Option 2: use Colab's secret storage (preferred; key not visible in notebook)
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```
The code reads the key from the environment at runtime; nothing is committed to git.

## Extending or tuning
- Adjust `max_expansions`, `max_subqueries`, or `model_name` when constructing `QueryTransformer`.
- Modify `_build_prompt` in `src/query/query_transformer.py` to change formatting or add domain-specific hints.
