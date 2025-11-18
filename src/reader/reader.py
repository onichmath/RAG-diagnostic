def create_rag_prompt(question: str, contexts: List[str]) -> str:
    """
    Create a RAG prompt with question and retrieved contexts.
    
    Args:
        question: User question
        contexts: Retrieved context documents
        
    Returns:
        Formatted prompt for the LLM
    """
    context_text = "\n\n".join([
        f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
    ])
    
    return f"""You are a medical assistant. Answer the following question based on the provided medical contexts.

Contexts:
{context_text}
Question: {question}
Answer:"""

class Generator:
    """Wrapper for qwen2.5:0.5b generation model."""
    
    def __init__(self, model_name, use_ollama: bool = True):
        """
        Initialize the generator.
        
        Args:
            model_name: Model identifier (for Ollama: "qwen2.5:0.5b", for HF: "Qwen/Qwen2.5-0.5B")
            use_ollama: If True, use Ollama API; if False, use HuggingFace transformers
        """
        if model_name is None or model_name == "":
            if use_ollama:
                model_name = "qwen2.5:0.5b"
            else:
                model_name = "Qwen/Qwen2.5-0.5B"
                
        self.model_name = model_name
        self.use_ollama = use_ollama
        
        if use_ollama:
            try:
                import ollama
                self.client = ollama
            except ImportError:
                raise ImportError(
                    "Ollama not installed. Install with: pip install ollama\n"
                    "Or set use_ollama=False to use HuggingFace transformers"
                )
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            except ImportError:
                raise ImportError(
                    "Transformers not installed. Install with: pip install transformers torch"
                )
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
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
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
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
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt):].strip()
        
    