"""
LLM-based reranking of retrieved documents using only their titles.
This version runs fully locally using HuggingFace Transformers (NO API KEYS) for Google Colab.
"""

import json
from typing import List
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Choose a model that works in Colab:
# On GPU Colab: Llama 3-8B works reasonably well.
# On CPU-only Colab: use microsoft/phi-2 (lighter).
# _MODEL_NAME = "microsoft/phi-2"
_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# Load tokenizer/model once at import time
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

# Set pad_token if not already set
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token

_model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
)


# Export for reuse in other modules
def get_shared_model():
    """Get the shared model instance."""
    return _model


def get_shared_tokenizer():
    """Get the shared tokenizer instance."""
    return _tokenizer


def get_shared_model_name():
    """Get the shared model name."""
    return _MODEL_NAME


def _build_prompt(query: str, titles: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

    prompt = f"""
You are a medical information retrieval assistant.

Rank the following DOCUMENT TITLES from MOST relevant to LEAST relevant
to the given QUESTION.

QUESTION:
{query}

DOCUMENT TITLES:
{numbered}

Return ONLY a JSON list of indices like:
[2, 1, 3]
"""
    return prompt.strip()


def rerank_by_title_llm(
    query: str,
    docs: List[Document],
    model_name: str = None,  # ignored, for compatibility
) -> List[Document]:

    if len(docs) <= 1:
        return docs

    titles = [
        (doc.metadata.get("title") or "").strip() or f"doc-{i}"
        for i, doc in enumerate(docs)
    ]

    prompt = _build_prompt(query, titles)

    # Tokenize + generate locally
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON array
    try:
        start = response.index("[")
        end = response.index("]") + 1
        json_str = response[start:end]

        ranked_indices = json.loads(json_str)
        ranked_zero_based = [i - 1 for i in ranked_indices]
        ranked_zero_based = [i for i in ranked_zero_based if 0 <= i < len(docs)]

        # reorder results
        reranked = [docs[i] for i in ranked_zero_based]

        # append missing
        missing = [i for i in range(len(docs)) if i not in ranked_zero_based]
        reranked.extend([docs[i] for i in missing])

        return reranked

    except Exception as e:
        print(f"[LLM RERANK WARNING] Failed to parse model output: {e}")
        print("Raw output:", response)
        return docs
