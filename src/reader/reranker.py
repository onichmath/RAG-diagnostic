from typing import List, Any, Protocol, Callable 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class Reranker(Protocol):
    """Protocol for reranking methods."""
    def rerank(self, query: str, results: List[Any], k: int) -> List[Any]:
        """
        Rerank search results.
        
        Args:
            query: The search query
            results: Initial search results from vectorstore
            k: Number of results to return after reranking
            
        Returns:
            Reranked list of results
        """
        ...


class NoReranker:
    """No reranking - returns results as-is."""
    def rerank(self, query: str, results: List[Any], k: int) -> List[Any]:
        return results[:k]
    
class HFReranker:
    """HF cross-encoder reranker.
    
    Expected model name: "bert-base-uncased"
    
    Args:
        model_name: Name of the HF model to use
        text_getter: Function to get text from results
        
    """
    def __init__(self, model_name: str, text_getter: Callable[[Any], str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        self.text_getter = text_getter or (lambda x: str(x))
        
    @torch.no_grad()
    def rerank(self, query: str, results: List[Any], k: int) -> List[Any]:
        if not results:
            return []
        
        docs = [self.text_getter(result) for result in results] 
        
        encoded_docs = self.tokenizer(
            [query] + docs,
            docs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        encoded_docs = {k: v.to(self.device) for k, v in encoded_docs.items()}
        
        outputs = self.model(**encoded_docs)
        
        logits = outputs.logits.squeeze(-1) # B
        
        scores = logits.detach().cpu().tolist()
        ranked_indices = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)

        k = min(k, len(results))
        top_indices = ranked_indices[:k]

        reranked_results = [results[i] for i in top_indices]
        return reranked_results