from typing import List, Dict, Optional, Any, Union
import json
import re
import os
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

# Try to import RAGAS llm_factory for OpenAI support
try:
    from ragas.llms import llm_factory

    _RAGAS_LLM_FACTORY_AVAILABLE = True
except ImportError:
    _RAGAS_LLM_FACTORY_AVAILABLE = False

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

# Import RAGAS embeddings - required for RAGAS metrics
try:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

    _RAGAS_EMBEDDINGS_AVAILABLE = True
    _RAGAS_EMBEDDINGS_IMPORT_ERROR = None
except ImportError as e:
    _RAGAS_EMBEDDINGS_AVAILABLE = False
    RagasHuggingFaceEmbeddings = None
    _RAGAS_EMBEDDINGS_IMPORT_ERROR = e

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


class JSONSanitizingLLM(HuggingFacePipeline):
    """
    A wrapper around HuggingFacePipeline that sanitizes the output to ensure it's valid JSON.
    This is necessary because some models (like Qwen) may add prefixes or other non-JSON text.
    """

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        raw_output = super()._call(prompt, stop=stop, **kwargs)

        # Attempt to extract JSON from the raw output
        json_str = self._extract_json(raw_output)

        if json_str:
            return json_str
        else:
            # If no valid JSON is found, return the raw output (RAGAS will likely error)
            print(
                f"[JSONSanitizingLLM WARNING] No valid JSON found in output. Raw: {raw_output[:200]}..."
            )
            return raw_output

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract valid JSON from model output, handling common formatting issues.

        Handles cases like:
        - "1. {...}" -> "{...}"
        - "1``` {...}" -> "{...}"
        - "1, {...}" -> "{...}"
        - "```json {...} ```" -> "{...}"
        """
        # Strategy 1: Find first '{' and last '}' or first '[' and last ']'
        # and try to parse the substring
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            try:
                start_idx = text.find(start_char)
                end_idx = text.rfind(end_char)
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    potential_json = text[start_idx : end_idx + 1]
                    json.loads(potential_json)  # Validate
                    return potential_json
            except json.JSONDecodeError:
                pass

        # Strategy 2: Regex to find common JSON patterns (e.g., inside ```json ... ```)
        match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
        if match:
            try:
                json_content = match.group(1)
                json.loads(json_content)  # Validate
                return json_content
            except json.JSONDecodeError:
                pass

        # Strategy 3: Remove common prefixes and try to parse
        # Examples: "1. {...", "1``` {...", "1, {..."
        cleaned_text = re.sub(
            r"^\s*(\d+\.?\s*|```\s*|\d+,\s*|```json\s*)", "", text, count=1
        )
        try:
            json.loads(cleaned_text)  # Validate
            return cleaned_text
        except json.JSONDecodeError:
            pass

        return None


def _build_llm(model_name: str = "local") -> Union[HuggingFacePipeline, Any]:
    """
    Create an LLM for Ragas (local HuggingFace or OpenAI API).

    Args:
        model_name:
            - "local" or HuggingFace model name (e.g., "vibrantlabsai/Ragas-critic-llm-Qwen1.5-GPTQ")
            - "openai:gpt-4" or "openai:gpt-4o" for OpenAI API (requires OPENAI_API_KEY env var)
            - "gpt-4" or "gpt-4o" (shortcut, automatically uses OpenAI)

    Returns:
        LLM instance (HuggingFacePipeline for local, RAGAS LLM for OpenAI)
    """
    # Handle OpenAI API models
    # Check for OpenAI models (but not "openai:local" which is invalid)
    is_openai_model = (
        model_name.startswith("openai:") and not model_name == "openai:local"
    ) or model_name in [
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    if is_openai_model:
        if not _RAGAS_LLM_FACTORY_AVAILABLE:
            raise ImportError(
                "RAGAS llm_factory not available. Install with: pip install ragas[openai]"
            )

        # Extract model name (remove "openai:" prefix if present)
        openai_model = model_name.replace("openai:", "")

        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )

        # Use cached model if available
        cache_key = f"openai:{openai_model}"
        if cache_key in _model_cache:
            return _model_cache[cache_key]

        print(f"Using OpenAI API model: {openai_model}...")

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            # Use RAGAS's llm_factory with OpenAI
            # Try with provider parameter first, fallback to auto-detect
            try:
                llm = llm_factory(openai_model, provider="openai", client=client)
            except (TypeError, ValueError):
                # Some RAGAS versions auto-detect provider from model name
                llm = llm_factory(openai_model, client=client)

            # Cache it
            _model_cache[cache_key] = llm

            return llm
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenAI LLM: {e}. "
                "Make sure OPENAI_API_KEY is set and valid."
            ) from e

    # Handle local HuggingFace models
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
            # max_new_tokens=2048,  # Increased for complete JSON generation
            do_sample=False,
            return_full_text=False,
        )

        # Wrap in JSONSanitizingLLM to handle JSON parsing issues
        llm = JSONSanitizingLLM(pipeline=pipe)

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
        # max_new_tokens=2048,  # Increased for complete JSON generation
        do_sample=False,
        return_full_text=False,
    )

    # Wrap in JSONSanitizingLLM to handle JSON parsing issues
    llm = JSONSanitizingLLM(pipeline=pipe)

    # Cache it
    _model_cache[model_name] = llm

    return llm


def _build_embeddings(embedding_model: str = None):
    """Create local HuggingFace embeddings for RAGAS metrics.

    RAGAS requires its own embeddings class, not LangChain's.
    """
    if embedding_model is None:
        embedding_model = _DEFAULT_EMBEDDING_MODEL

    # Use cached embeddings if available
    if embedding_model in _embedding_cache:
        return _embedding_cache[embedding_model]

    # Check if RAGAS embeddings are available
    if not _RAGAS_EMBEDDINGS_AVAILABLE or RagasHuggingFaceEmbeddings is None:
        error_msg = f"RAGAS embeddings not available."
        if _RAGAS_EMBEDDINGS_IMPORT_ERROR:
            error_msg += f" Import error: {_RAGAS_EMBEDDINGS_IMPORT_ERROR}."
        error_msg += " Make sure 'ragas' is properly installed: pip install ragas"
        raise RuntimeError(error_msg)

    # RAGAS requires its own embeddings class - don't fallback to LangChain
    # Try different parameter names that RAGAS might accept
    embeddings = None
    last_error = None

    # Try with 'model' parameter (most common)
    try:
        embeddings = RagasHuggingFaceEmbeddings(model=embedding_model)
    except (TypeError, AttributeError, ValueError) as e:
        last_error = e
        # Try with 'model_name' parameter
        try:
            embeddings = RagasHuggingFaceEmbeddings(model_name=embedding_model)
        except (TypeError, AttributeError, ValueError) as e2:
            last_error = e2
            # Try positional argument
            try:
                embeddings = RagasHuggingFaceEmbeddings(embedding_model)
            except Exception as e3:
                last_error = e3

    if embeddings is None:
        raise RuntimeError(
            f"Failed to create RAGAS embeddings with model '{embedding_model}'. "
            f"Error: {last_error}. "
            "RAGAS requires its own embeddings class. "
            "Tried parameters: 'model', 'model_name', and positional. "
            "Check RAGAS documentation for correct usage."
        )

    _embedding_cache[embedding_model] = embeddings
    return embeddings


def _select_metrics(
    metric_names: List[str],
    llm: Union[HuggingFacePipeline, Any],
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


def compute_ragas_metrics_batch(
    eval_data: List[Dict[str, Any]],
    model_name: str = "local",
    embedding_model: Optional[str] = None,
    metrics: Optional[List[str]] = [
        # "faithfulness",
        # "answer_relevancy",
        "context_precision",
        "context_recall",
    ],
) -> List[Dict[str, float]]:
    """
    Compute RAGAS metrics for multiple query-answer pairs in batch (much faster).

    Args:
        eval_data: List of dicts, each with keys: "question", "contexts", "answer", "ground_truth"
        model_name: Model name (see compute_ragas_metrics for options)
        embedding_model: Embedding model for RAGAS metrics
        metrics: Which metrics to compute

    Returns:
        List of dicts, each mapping metric name -> score (0.0–1.0)
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if not eval_data:
        return []

    llm = _build_llm(model_name)
    embeddings = _build_embeddings(embedding_model)
    selected_metrics = _select_metrics(metrics, llm, embeddings)

    if not selected_metrics:
        return [{}] * len(eval_data)

    # Create dataset with all queries at once
    dataset = Dataset.from_list(eval_data)

    # Evaluate all at once (much faster than one-by-one)
    result = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
    )

    # Extract scores for all queries
    all_scores = []

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        for idx in range(len(eval_data)):
            scores = {}
            for name in metrics:
                if name in df.columns:
                    scores[name] = float(df[name].iloc[idx])
                else:
                    raise ValueError(
                        f"Metric '{name}' not found in RAGAS results. Available columns: {df.columns.tolist()}"
                    )
            all_scores.append(scores)
    elif hasattr(result, "column_names"):
        for idx in range(len(eval_data)):
            scores = {}
            for name in metrics:
                if name in result.column_names:
                    value = result[name][idx]
                    scores[name] = float(value)
                else:
                    raise ValueError(
                        f"Metric '{name}' not found in RAGAS results. Available columns: {result.column_names}"
                    )
            all_scores.append(scores)
    elif hasattr(result, "__getitem__"):
        for idx in range(len(eval_data)):
            scores = {}
            for name in metrics:
                if name in result:
                    value = (
                        result[name][idx]
                        if isinstance(result[name], list)
                        else result[name]
                    )
                    scores[name] = float(value)
                else:
                    raise ValueError(f"Metric '{name}' not found in RAGAS results")
            all_scores.append(scores)
    else:
        # Fallback: assume single result per query
        for idx in range(len(eval_data)):
            scores = {}
            for name in metrics:
                if hasattr(result, name):
                    value = getattr(result, name)
                    if isinstance(value, (list, tuple)) and len(value) > idx:
                        scores[name] = float(value[idx])
                    else:
                        scores[name] = float(value) if idx == 0 else 0.0
                else:
                    raise ValueError(f"Metric '{name}' not found in RAGAS results")
            all_scores.append(scores)

    return all_scores


def compute_ragas_metrics(
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str,
    model_name: str = "local",
    embedding_model: Optional[str] = None,
    metrics: Optional[List[str]] = [
        # "faithfulness",
        # "answer_relevancy",
        "context_precision",
        "context_recall",
    ],
) -> Dict[str, float]:
    """
    Compute RAGAS metrics for a single query-answer pair.

    Args:
        question: User question.
        contexts: List of retrieved context strings for this question.
        answer: Generated answer from your RAG system.
        ground_truth: Gold answer (empty string if not available).
        model_name:
            - "local" (default) uses vibrantlabsai/Ragas-critic-llm-Qwen1.5-GPTQ
            - HuggingFace model name (e.g., "microsoft/phi-2")
            - "openai:gpt-4" or "gpt-4" for OpenAI API (requires OPENAI_API_KEY)
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
