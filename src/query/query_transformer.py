"""
Query transformation utilities built on google.colab.ai models with a fallback
to the Gemini API (google.genai) when Colab AI is unavailable.

Features:
- Query rewriting: normalize a clinical query into a concise, unambiguous form.
- Query expansion: produce alternate phrasings/synonyms to widen recall.
- Query decomposition: break complex questions into atomic sub-questions.
- Step-back prompting: add a higher-level version to pull broader context.

If neither provider is available, the transformer falls back to pass-through
behavior so retrieval still works.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import List, Optional
import os

logger = logging.getLogger(__name__)

# Lazy-loaded reference to google.colab.ai
# We delay the import to avoid kernel connection issues when other heavy
# imports (like HuggingFace models) happen between import and first call.
_ai_module = None
_ai_import_attempted = False
_genai_import_attempted = False
_genai_client = None


def _get_ai_module():
    """Lazy import of google.colab.ai to ensure kernel is ready at call time."""
    global _ai_module, _ai_import_attempted
    
    if _ai_import_attempted:
        return _ai_module
    
    _ai_import_attempted = True
    try:
        from google.colab import ai
        _ai_module = ai
        logger.info("google.colab.ai loaded successfully")
    except Exception as exc:
        logger.warning("google.colab.ai not available: %s", exc)
        _ai_module = None
    
    return _ai_module


def _get_genai_client():
    """
    Lazy import + client construction for google.genai (Gemini API).
    The client looks for GEMINI_API_KEY in the environment.
    """
    global _genai_import_attempted, _genai_client

    if _genai_import_attempted:
        return _genai_client

    _genai_import_attempted = True

    try:
        from google import genai  # type: ignore

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set; Gemini fallback disabled")
            _genai_client = None
            return None

        _genai_client = genai.Client(api_key=api_key)
        logger.info("google.genai client initialized for Gemini fallback")
    except Exception as exc:  # pragma: no cover - external dependency
        logger.warning("google.genai not available: %s", exc)
        _genai_client = None

    return _genai_client


@dataclass
class QueryTransformResult:
    """Container for the structured outputs of query transformation."""

    original: str
    rewrite: Optional[str] = None
    expansions: List[str] = field(default_factory=list)
    decompositions: List[str] = field(default_factory=list)
    step_back: Optional[str] = None

    def candidate_queries(self) -> List[str]:
        """
        Build a deduplicated list of queries to issue against the retriever.

        Order favors the rewritten query, then broader/alternate forms to
        increase recall, while keeping the list compact.
        """
        ordered = [
            self.rewrite or self.original,
            self.step_back,
            *self.expansions,
            *self.decompositions,
        ]
        seen = set()
        result = []
        for q in ordered:
            q_clean = (q or "").strip()
            if not q_clean:
                continue
            key = q_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(q_clean)

        # Always ensure the original query is present.
        if self.original.strip() and self.original.strip().lower() not in seen:
            result.insert(0, self.original.strip())
        return result


class QueryTransformer:
    """
    Generates multiple query variants using google.colab.ai, with a Gemini
    fallback when Colab AI is not available.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        max_expansions: int = 5,
        max_subqueries: int = 3,
    ):
        # Store raw model name (without provider prefix)
        # Handles both "gemini-2.5-flash" and "google/gemini-2.5-flash" formats
        self.model_name = self._extract_model_name(model_name)
        self.max_expansions = max_expansions
        self.max_subqueries = max_subqueries
        # Note: availability is checked lazily via _get_ai_module/_get_genai_client

    @staticmethod
    def _extract_model_name(model_name: str) -> str:
        """
        Extract the base model name from provider/model format.
        
        Examples:
            "google/gemini-2.5-flash" -> "gemini-2.5-flash"
            "gemini-2.5-flash" -> "gemini-2.5-flash"
            "openai/gpt-4" -> "gpt-4"
        """
        if "/" in model_name:
            return model_name.split("/", 1)[1]
        return model_name

    def transform(self, query: str) -> QueryTransformResult:
        """
        Run the transformation stack. Falls back to identity if both providers are absent.
        """
        prompt = self._build_prompt(query)
        raw = self._call_model(prompt)
        parsed = self._parse_response(raw, query)

        return parsed

    def _build_prompt(self, query: str) -> str:
        """
        Craft a structured prompt asking the model to return JSON with all
        transformation pieces in one shot to limit round trips.
        """
        return f"""
You are assisting a clinical retrieval system. Given the user query, produce a JSON object with four fields:
- "rewrite": one concise, professional reformulation that preserves intent.
- "expansions": up to {self.max_expansions} alternate phrasings or key synonyms to improve recall.
- "decompositions": up to {self.max_subqueries} atomic sub-questions that together answer the original.
- "step_back": a broader, high-level version to fetch contextual guidance.

Keep answers short (<=20 words each). Return ONLY JSON, no prose.
User query: "{query}"
"""

    def _call_model(self, prompt: str) -> str:
        """
        Try Colab AI first; if unavailable or failing, fall back to Gemini API.
        """
        # Attempt Colab AI
        ai = _get_ai_module()
        if ai is not None:
            text = self._call_colab_ai(prompt, ai)
            if text:
                return text

        # Attempt Gemini API
        client = _get_genai_client()
        if client is not None:
            text = self._call_gemini(prompt, client)
            if text:
                return text

        # Final fallback: empty string (caller will pass through original query)
        return ""

    def _call_colab_ai(self, prompt: str, ai) -> str:
        """Invoke google.colab.ai.
        
        Note: Colab AI expects model_name in "provider/model" format (e.g., "google/gemini-2.5-flash")
        """
        try:
            # Colab AI requires the "google/" prefix for Gemini models
            colab_model_name = f"google/{self.model_name}"
            return ai.generate_text(prompt, model_name=colab_model_name)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Query transform generation failed via Colab AI: %s", exc)
            return ""

    def _call_gemini(self, prompt: str, client) -> str:
        """Invoke the Gemini API through google.genai.
        
        Note: google.genai expects model in "models/model-name" or "model-name" format,
        NOT in "provider/model" format like Colab AI.
        """
        try:
            # google.genai can use just the model name (e.g., "gemini-2.5-flash")
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            # Some client versions wrap text in .text; others in .candidates
            if hasattr(response, "text"):
                return response.text or ""
            if hasattr(response, "candidates") and response.candidates:
                return getattr(response.candidates[0], "text", "") or ""
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Query transform generation failed via Gemini API: %s", exc)
        return ""

    def _parse_response(
        self,
        response: str,
        original_query: str,
    ) -> QueryTransformResult:
        """
        Parse the model output into a QueryTransformResult, tolerating imperfect
        JSON by applying simple recovery heuristics.
        """
        rewrite = None
        expansions: List[str] = []
        decompositions: List[str] = []
        step_back = None

        if response:
            try:
                # Extract JSON portion if extra text is present.
                start = response.find("{")
                end = response.rfind("}") + 1
                json_blob = response[start:end] if start != -1 else response
                payload = json.loads(json_blob)
                rewrite = self._clean_str(payload.get("rewrite"))
                expansions = self._clean_list(payload.get("expansions"))
                decompositions = self._clean_list(payload.get("decompositions"))
                step_back = self._clean_str(payload.get("step_back"))
            except Exception as exc:
                logger.warning("Failed to parse query transform output: %s", exc)

        # Ensure sensible defaults when parsing fails.
        rewrite = rewrite or original_query
        return QueryTransformResult(
            original=original_query,
            rewrite=rewrite,
            expansions=expansions,
            decompositions=decompositions,
            step_back=step_back,
        )

    @staticmethod
    def _clean_str(value: Optional[str]) -> Optional[str]:
        """Strip whitespace and ignore empty strings."""
        if not value:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _clean_list(values: Optional[List[str]]) -> List[str]:
        """Normalize list fields to a compact list of non-empty strings."""
        if not values:
            return []
        cleaned = []
        seen = set()
        for v in values:
            v_clean = (v or "").strip()
            if not v_clean:
                continue
            key = v_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(v_clean)
        return cleaned
