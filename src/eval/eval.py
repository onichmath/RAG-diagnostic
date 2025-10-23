from pathlib import Path
from src.retriever.faiss_builder import load_faiss_index
from time import time
import json

def load_test_queries(queries_file: Path, max_queries: int):
    """
    Load the test queries from the given file.
    """
    with open(queries_file, 'r') as f:
        data = json.load(f)
    return data.get('queries', [])[:max_queries]

def evaluate_rag_system(index_path: Path, queries_file: Path, max_queries: int, k_array: list):
    """
    Evaluate the RAG system using the given index and queries.
    """
    test_queries = load_test_queries(queries_file, max_queries)
    vectorstore = load_faiss_index(index_path) 
    results = {}
    
    for k in k_array:
        total_latency_at_k = 0.0
        total_precision_at_k = 0.0
        total_ndcg_at_k = 0.0
        total_queries = len(test_queries)
        
        for i, query_data in enumerate(test_queries, 1):
            query_text = query_data.get('query_text', '')
            expected_source = "guideline"
            
            time_start = time()
            search_results = vectorstore.similarity_search(query_text, k=k)
            time_end = time()
            latency = time_end - time_start
                                    
            relevant_found = 0
            relevant_positions = []
            relevance_scores = [] 
            
            for i, result in enumerate(search_results):
                is_relevant = False
                
                if result.metadata.get('source') == expected_source:
                    is_relevant = True
                    break
                
                if is_relevant:
                    relevant_found += 1
                    relevant_positions.append(i + 1)
                    relevance_scores.append(1.0)
                else:
                    relevance_scores.append(0.0)
            
            precision_at_k = relevant_found / k if k > 0 else 0.0
            
            def dcg_at_k(relevance_scores, k):
                """Calculate DCG@k"""
                dcg = 0.0
                for i in range(min(k, len(relevance_scores))):
                    dcg += relevance_scores[i] / (i + 1) if relevance_scores[i] > 0 else 0.0
                return dcg
            
            def idcg_at_k(relevance_scores, k):
                """Calculate IDCG@k (ideal DCG)"""
                sorted_scores = sorted(relevance_scores, reverse=True)
                return dcg_at_k(sorted_scores, k)
            
            dcg = dcg_at_k(relevance_scores, k)
            idcg = idcg_at_k(relevance_scores, k)
            ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
            
            total_precision_at_k += precision_at_k
            total_ndcg_at_k += ndcg_at_k
            total_latency_at_k += latency 
        
        total_time = total_latency_at_k
        throughput = total_queries / total_time if total_time > 0 else 0.0 # QPS
        
        results[k] = {
            "avg_precision_at_k": total_precision_at_k / total_queries,
            "avg_ndcg_at_k": total_ndcg_at_k / total_queries,
            "avg_latency_ms": (total_latency_at_k / total_queries) * 1000,
            "throughput_qps": throughput,
            "total_queries": total_queries
        }
    
    return results

def save_results(results: dict, output_file: Path):
    """
    Save the results to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f)