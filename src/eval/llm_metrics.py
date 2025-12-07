from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

# Default model - same as llm_title_rerank.py
_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# Default embedding model for RAGAS metrics
_DEFAULT_EMBEDDING_MODEL = "thenlper/gte-small"

# Global model cache
_model_cache = {}
_embedding_cache = {}

# Try to import shared model from llm_title_rerank
try:
    from src.reranker.llm_title_rerank import (
        get_shared_model,
        get_shared_tokenizer,
        get_shared_model_name,
    )

    _SHARED_MODEL_AVAILABLE = True
except ImportError:
    _SHARED_MODEL_AVAILABLE = False


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def _build_llm(model_name: str = "local") -> HuggingFacePipeline:
    """
    Create a local HuggingFace LLM for Ragas, reusing model from llm_title_rerank.py if available.

    Args:
        model_name: Model name or "local" to use default (microsoft/phi-2)

    Returns:
        HuggingFacePipeline LLM instance
    """
    if model_name == "local":
        model_name = _DEFAULT_MODEL_NAME

    # Use cached model if available
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Try to reuse shared model from llm_title_rerank if available and matching
    if _SHARED_MODEL_AVAILABLE and model_name == get_shared_model_name():
        print(f"Reusing shared model from llm_title_rerank: {model_name}...")
        shared_model = get_shared_model()
        shared_tokenizer = get_shared_tokenizer()

        # Create pipeline using shared model
        pipe = pipeline(
            "text-generation",
            model=shared_model,
            tokenizer=shared_tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=0,
            return_full_text=False,
        )

        # Wrap in LangChain HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)

        # Cache it
        _model_cache[model_name] = llm

        return llm

    # Otherwise, load new model
    print(f"Loading local model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=0,
        return_full_text=False,
    )

    # Wrap in LangChain HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    # Cache it
    _model_cache[model_name] = llm

    return llm


def _build_embeddings(embedding_model: str = None):
    """Create local HuggingFace embeddings for RAGAS metrics."""
    if embedding_model is None:
        embedding_model = _DEFAULT_EMBEDDING_MODEL

    # Use cached embeddings if available
    if embedding_model in _embedding_cache:
        return _embedding_cache[embedding_model]

    # Use RAGAS's HuggingFace embeddings wrapper
    try:
        embeddings = RagasHuggingFaceEmbeddings(model=embedding_model)
    except (ImportError, AttributeError):
        # Fallback to LangChain embeddings if RAGAS wrapper not available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs={"device": device}
        )

    _embedding_cache[embedding_model] = embeddings
    return embeddings


def _select_metrics(
    metric_names: List[str],
    llm: HuggingFacePipeline,
    embeddings,
):
    """Map metric name strings to Ragas metric objects and attach LLM/embeddings if needed."""
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    selected = []
    for name in metric_names:
        # Create new metric instances to avoid shared state
        if name == "faithfulness":
            metric = faithfulness(llm=llm)
        elif name == "answer_relevancy":
            metric = answer_relevancy(llm=llm, embeddings=embeddings)
        elif name == "context_precision":
            metric = context_precision(embeddings=embeddings)
        elif name == "context_recall":
            metric = context_recall(embeddings=embeddings)
        else:
            continue
        if hasattr(metric, "llm"):
            metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

        selected.append(metric)

    return selected


def compute_ragas_metrics(
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str,
    model_name: str = "local",
    metrics: Optional[List[str]] = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ],
) -> Dict[str, float]:
    """
    Compute RAGAS metrics for a single query-answer pair using local HuggingFace model.

    Args:
        question: User question.
        contexts: List of retrieved context strings for this question.
        answer: Generated answer from your RAG system.
        ground_truth: Gold answer (empty string if not available).
        model_name: Local model name (default: "local" uses microsoft/phi-2).
                    Can also specify any HuggingFace model name.
        metrics: Which metrics to compute. Defaults to a standard set:
                 ["faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"].

    Returns:
        Dict mapping metric name -> score (0.0–1.0).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    llm = _build_llm(model_name)
    embeddings = _build_embeddings()
    selected_metrics = _select_metrics(metrics, llm, embeddings)

    if not selected_metrics:
        return {}

    eval_row = {
        "question": question,
        "contexts": contexts,  # list[str] for this single example
        "answer": answer,
        "ground_truth": ground_truth or "",
    }
    dataset = Dataset.from_list([eval_row])

    result = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
    )

    scores: Dict[str, float] = {}
    for name in metrics:
        if name in result.column_names:
            value = result[name][0]
            try:
                scores[name] = float(value)
            except (TypeError, ValueError):
                scores[name] = -1.0

    return scores
