from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retriever.faiss_builder import load_faiss_index
from time import time
import json
import math
from matplotlib import pyplot as plt
import seaborn as sns

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
    # Warm start 
    for i in range(1):
        vectorstore.similarity_search("test", k=10)
    
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
                if (result.metadata.get('source') == expected_source):
                    relevant_found += 1
                    relevant_positions.append(i + 1)
                    relevance_scores.append(1.0)
                else:
                    relevance_scores.append(0.0)
            
            precision_at_k = relevant_found / k if k > 0 else 0.0
            
            def dcg_at_k(relevance_scores, k):
                """Calculate DCG@k with proper log scaling"""
                dcg = 0.0
                for i in range(min(k, len(relevance_scores))):
                    if relevance_scores[i] > 0:
                        dcg += relevance_scores[i] / math.log2(i + 2)
                return dcg
            
            def idcg_at_k(relevance_scores, k):
                """Calculate IDCG@k (ideal DCG)"""
                total_relevant = sum(relevance_scores)
                ideal_scores = [1.0] * min(int(total_relevant), k) + [0.0] * max(0, k - int(total_relevant))
                return dcg_at_k(ideal_scores, k)
            
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
        
        
def graph_results(results: dict, graph_file: Path):
    """
    Graph the results with precision@k and NDCG@k.
    """
    # TODO 
    raise NotImplementedError("Graphing results is not implemented yet")
    
def load_results(results_file: Path):
    """
    Load the results from a JSON file.
    """
    with open(results_file, 'r') as f:
        return json.load(f)
    
if __name__ == "__main__":
    results = load_results(Path("data/FAISS_evaluation_results.json"))
    graph_results(results, Path("data/FAISS_evaluation_results.png"))
    exit()