from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Suppress torchvision beta warnings
try:
    import torchvision

    torchvision.disable_beta_transforms_warning()
except ImportError:
    pass

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Try to use updated langchain-huggingface package, fallback to deprecated version
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline
    import warnings

    warnings.warn(
        "Using deprecated langchain_community.llms.HuggingFacePipeline. "
        "Install langchain-huggingface for the updated version.",
        DeprecationWarning,
        stacklevel=2,
    )

from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

# Default model for RAGAS judge (optimized for RAGAS evaluation)
_DEFAULT_MODEL_NAME = "vibrantlabsai/Ragas-critic-llm-Qwen1.5-GPTQ"
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
    Create a local HuggingFace LLM for Ragas.
    Uses a different default model than the LLM reranker (optimized for RAGAS).

    Args:
        model_name: Model name or "local" to use default (vibrantlabsai/Ragas-critic-llm-Qwen1.5-GPTQ)

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

        # Set pad_token if not already set
        if shared_tokenizer.pad_token is None:
            shared_tokenizer.pad_token = shared_tokenizer.eos_token

        # Create pipeline using shared model
        pipe = pipeline(
            "text-generation",
            model=shared_model,
            tokenizer=shared_tokenizer,
            max_new_tokens=2048,
            do_sample=False,
            return_full_text=False,
        )

        # Wrap in LangChain HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)

        # Cache it
        _model_cache[model_name] = llm

        return llm

    # Otherwise, load new model
    print(f"Loading RAGAS judge model: {model_name}...")

    # GPTQ models need trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GPTQ models typically use float16 and need trust_remote_code
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # GPTQ models use float16
        device_map="auto",
        trust_remote_code=True,
    )

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,  # Increased for complete JSON generation
        do_sample=False,
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
        metric = metric_map.get(name)
        if metric is None:
            continue

        # Metrics are already instantiated objects, just set attributes directly
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
    embedding_model: Optional[str] = None,
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
        model_name: Local model name (default: "local" uses vibrantlabsai/Ragas-critic-llm-Qwen1.5-GPTQ).
                    Can also specify any HuggingFace model name.
        embedding_model: Embedding model for RAGAS metrics (default: uses _DEFAULT_EMBEDDING_MODEL).
        metrics: Which metrics to compute. Defaults to a standard set:
                 ["faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"].

    Returns:
        Dict mapping metric name -> score (0.0–1.0).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    llm = _build_llm(model_name)
    embeddings = _build_embeddings(embedding_model)
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

    # RAGAS has a default timeout of 180 seconds (3 minutes) per metric
    # If your model is slow, you may need to set RAGAS_TIMEOUT environment variable
    # or use a faster model (e.g., GPTQ quantized models)
    result = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
    )

    scores: Dict[str, float] = {}

    # Handle different RAGAS result types
    # Newer versions return EvaluationResult, older versions return Dataset
    if hasattr(result, "to_pandas"):
        # EvaluationResult - convert to pandas DataFrame
        df = result.to_pandas()
        for name in metrics:
            if name in df.columns:
                scores[name] = float(df[name].iloc[0])
            else:
                raise ValueError(
                    f"Metric '{name}' not found in RAGAS results. Available columns: {df.columns.tolist()}"
                )
    elif hasattr(result, "column_names"):
        # Dataset with column_names attribute
        for name in metrics:
            if name in result.column_names:
                value = result[name][0]
                scores[name] = float(value)
            else:
                raise ValueError(
                    f"Metric '{name}' not found in RAGAS results. Available columns: {result.column_names}"
                )
    elif hasattr(result, "__getitem__"):
        # Dataset-like object
        for name in metrics:
            if name in result:
                value = (
                    result[name][0] if isinstance(result[name], list) else result[name]
                )
                scores[name] = float(value)
            else:
                raise ValueError(f"Metric '{name}' not found in RAGAS results")
    else:
        # Try to access as dict
        for name in metrics:
            if hasattr(result, name):
                value = getattr(result, name)
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    scores[name] = float(value[0])
                else:
                    scores[name] = float(value)
            else:
                raise ValueError(f"Metric '{name}' not found in RAGAS results")

    return scores
