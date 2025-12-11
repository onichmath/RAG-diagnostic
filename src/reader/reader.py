from typing import List, Optional
import torch


def create_rag_prompt(question: str, contexts: List[str]) -> str:
    """
    Create a RAG prompt with question and retrieved contexts.

    Args:
        question: User question
        contexts: Retrieved context documents

    Returns:
        Formatted prompt for the LLM
    """
    context_text = "\n\n".join(
        [f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)]
    )

    return f"""You are a medical assistant. Answer the following question based on the provided medical contexts.

Contexts:
{context_text}
Question: {question}
Answer:"""


class Generator:
    """Wrapper for microsoft/phi-2 generation model."""

    def __init__(
        self,
        model_name=None,
        use_ollama: bool = False,
        shared_model=None,
        shared_tokenizer=None,
    ):
        """
        Initialize the generator.

        Args:
            model_name: Model identifier (ignored if shared_model provided)
            use_ollama: If True, use Ollama API; if False, use HuggingFace transformers
            shared_model: Optional pre-loaded model to reuse (from llm_title_rerank)
            shared_tokenizer: Optional pre-loaded tokenizer to reuse (from llm_title_rerank)
        """
        self.use_ollama = use_ollama

        if use_ollama:
            if model_name is None or model_name == "":
                model_name = "phi"
            self.model_name = model_name
            try:
                import ollama

                self.client = ollama
            except ImportError:
                raise ImportError(
                    "Ollama not installed. Install with: pip install ollama\n"
                    "Or set use_ollama=False to use HuggingFace transformers"
                )
        else:
            if shared_model is not None and shared_tokenizer is not None:
                self.model = shared_model
                self.tokenizer = shared_tokenizer
                self.device = next(self.model.parameters()).device
                self.model_name = "shared_model"
            else:
                if model_name is None or model_name == "":
                    model_name = "microsoft/phi-2"
                self.model_name = model_name
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch

                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(
                            torch.float16 if self.device == "cuda" else torch.float32
                        ),
                        device_map="auto" if self.device == "cuda" else None,
                    )
                    if self.device == "cpu":
                        self.model = self.model.to(self.device)
                except ImportError:
                    raise ImportError(
                        "Transformers not installed. Install with: pip install transformers torch"
                    )

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if self.use_ollama:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": max_tokens, "temperature": temperature},
            )
            return response.get("response", "")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt) :].strip()
