"""
ColBERT-style (Late Interaction) reranker.
Computes the MaxSim score between query tokens and document tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from langchain_core.documents import Document

# Global cache to prevent reloading model on every function call
_tokenizer = None
_model = None
_device = None

def _get_model(model_name: str):
    """Singleton loader for the model."""
    global _tokenizer, _model, _device
    
    if _model is None:
        print(f"[ColBERT] Loading model: {model_name}...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
            
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.to(_device)
        _model.eval()
        
    return _tokenizer, _model, _device

def compute_colbert_score(query_enc, doc_enc):
    """
    Core ColBERT logic: Sum of Maximum Similarities (MaxSim).
    1. Calculate similarity matrix (Query Tokens x Doc Tokens)
    2. For every Query Token, find the max similarity across all Doc Tokens.
    3. Sum those max scores.
    """
    # query_enc: (batch, q_len, dim)
    # doc_enc:   (batch, d_len, dim)
    
    # 1. Dot product (similarity matrix)
    # Result: (batch, q_len, d_len)
    scores = torch.matmul(query_enc, doc_enc.transpose(-2, -1))
    
    # 2. Max over document tokens (dim=-1)
    # Result: (batch, q_len)
    max_scores = torch.max(scores, dim=-1).values
    
    # 3. Sum over query tokens
    # Result: (batch,)
    total_score = torch.sum(max_scores, dim=-1)
    
    return total_score.item()

def rerank_with_colbert(
    query: str,
    docs: List[Document],
    model_name: str = "bert-base-uncased",
    top_k: int = 5
) -> List[Document]:
    """
    Rerank a list of documents using ColBERT-style late interaction.
    """
    if not docs:
        return []

    tokenizer, model, device = _get_model(model_name)

    # 1. Tokenize & Embed Query
    q_inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        q_out = model(**q_inputs).last_hidden_state # (1, q_len, dim)
        # Normalize for cosine similarity
        q_embeds = F.normalize(q_out, p=2, dim=-1)

    scored_docs = []

    # 2. Loop through documents (Simple loop for clarity, can be batched later)
    for doc in docs:
        # Use page_content for the actual text
        content = doc.page_content
        
        d_inputs = tokenizer(content, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            d_out = model(**d_inputs).last_hidden_state # (1, d_len, dim)
            # Normalize
            d_embeds = F.normalize(d_out, p=2, dim=-1)
        
        # 3. Compute Score
        score = compute_colbert_score(q_embeds, d_embeds)
        
        # Store tuple: (score, doc_object)
        scored_docs.append((score, doc))

    # 4. Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 5. Return top_k Document objects
    # Note: We return the original Document objects, now reordered
    final_docs = [doc for score, doc in scored_docs[:top_k]]
    
    return final_docs