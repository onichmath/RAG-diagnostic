from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retriever.faiss_builder import load_faiss_index
from src.reranker.llm_title_rerank import rerank_by_title_llm
from src.reader.reader import Generator, create_rag_prompt
from src.eval.llm_metrics import compute_ragas_metrics

from time import time
import json
import math
from typing import Optional
from matplotlib import pyplot as plt
import seaborn as sns


def load_test_queries(queries_file: Path, max_queries: int):
    """
    Load the test queries from the given file.
    """
    with open(queries_file, "r") as f:
        data = json.load(f)
    return data.get("queries", [])[:max_queries]


def get_expected_doc_patterns(expected_docs: list):
    expected_doc_patterns = []
    for expected_doc in expected_docs:
        if expected_doc == "idsa_clinical_guideline_covid19":
            expected_doc_patterns.extend(
                ["idsa", "covid", "covid19", "clinical_guideline"]
            )
        elif expected_doc == "ada_soc_diabetes_2024":
            expected_doc_patterns.extend(["ada", "diabetes", "soc_diabetes"])
        elif expected_doc == "aha_stroke_2021":
            expected_doc_patterns.extend(["aha", "stroke", "stroke_2021"])
        elif expected_doc == "aha_acc_afib":
            expected_doc_patterns.extend(["aha", "acc", "afib", "atrial"])
        elif expected_doc == "acc_aha_hf":
            expected_doc_patterns.extend(["acc", "aha", "hf", "heart_failure"])
        elif expected_doc == "surviving_sepsis":
            expected_doc_patterns.extend(["surviving", "sepsis", "septic"])
        else:
            expected_doc_patterns.append(expected_doc.lower())
    return expected_doc_patterns


# def evaluate_rag_system(index_path: Path, queries_file: Path, max_queries: int, k_array: list):
def evaluate_rag_system(
    index_path: Path,
    queries_file: Path,
    max_queries: int,
    k_array: list,
    use_llm_reranker: bool = False,
    llm_model: str = "local",
    generator: Optional[Generator] = None,
    use_ragas: bool = False,
    ragas_model: str = "local",
):
    """
    Evaluate the RAG system using the given index and queries.

    Args:
        index_path: Path to FAISS index
        queries_file: Path to queries JSON file
        max_queries: Maximum number of queries to evaluate
        k_array: List of k values to evaluate
        use_llm_reranker: Whether to use LLM-based reranking
        llm_model: Model name for reranker (default: "local")
        generator: Optional Generator instance for answer generation (required for RAGAS)
        use_ragas: Whether to compute RAGAS metrics (requires generator)
        ragas_model: Model name for RAGAS judge (default: "local")
    """
    test_queries = load_test_queries(queries_file, max_queries)
    vectorstore = load_faiss_index(index_path)
    # Warm start
    for i in range(1):
        vectorstore.similarity_search("test", k=10)

    if use_ragas and generator is None:
        raise ValueError("Generator is required when use_ragas=True")

    results = {}

    for k in k_array:
        total_latency_at_k = 0.0
        total_precision_at_k = 0.0
        total_ndcg_at_k = 0.0
        total_queries = len(test_queries)

        # RAGAS metrics totals
        ragas_totals = {}

        for i, query_data in enumerate(test_queries, 1):
            query_text = query_data.get("query_text", "")
            expected_gold_docs = query_data.get("expected_gold_docs", [])
            golden_answer = query_data.get("golden_answer", "")

            expected_source = "guideline"

            time_start = time()
            search_results = vectorstore.similarity_search(query_text, k=k)
            # Title-Based Local LLM Reranker
            if use_llm_reranker:
                search_results = rerank_by_title_llm(
                    query=query_text,
                    docs=search_results,
                    model_name=llm_model,
                )
            time_end = time()
            latency = time_end - time_start

            # Extract contexts for RAGAS
            contexts = [doc.page_content for doc in search_results]

            # Generate answer and compute RAGAS metrics if requested
            if use_ragas and generator:
                prompt = create_rag_prompt(query_text, contexts)
                answer = generator.generate(prompt, max_tokens=512, temperature=0.7)

                ragas_scores = compute_ragas_metrics(
                    question=query_text,
                    contexts=contexts,
                    answer=answer,
                    ground_truth=golden_answer,
                    model_name=ragas_model,
                )

                # Accumulate RAGAS metrics
                for metric_name, score in ragas_scores.items():
                    if metric_name not in ragas_totals:
                        ragas_totals[metric_name] = 0.0
                    ragas_totals[metric_name] += score

            relevant_found = 0
            relevant_positions = []
            relevance_scores = []

            for i, result in enumerate(search_results):
                result_title = result.metadata.get("title", "").lower()
                expected_doc_patterns = get_expected_doc_patterns(expected_gold_docs)
                pattern_match = any(
                    pattern in result_title for pattern in expected_doc_patterns
                )
                result_source = result.metadata.get("source", "").lower()

                # if (result.metadata.get('source') == expected_source):
                if pattern_match and result_source == expected_source:
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
                ideal_scores = [1.0] * min(int(total_relevant), k) + [0.0] * max(
                    0, k - int(total_relevant)
                )
                return dcg_at_k(ideal_scores, k)

            dcg = dcg_at_k(relevance_scores, k)
            idcg = idcg_at_k(relevance_scores, k)
            ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

            total_precision_at_k += precision_at_k
            total_ndcg_at_k += ndcg_at_k
            total_latency_at_k += latency

        total_time = total_latency_at_k
        throughput = total_queries / total_time if total_time > 0 else 0.0  # QPS

        results[k] = {
            "avg_precision_at_k": total_precision_at_k / total_queries,
            "avg_ndcg_at_k": total_ndcg_at_k / total_queries,
            "avg_latency_ms": (total_latency_at_k / total_queries) * 1000,
            "throughput_qps": throughput,
            "total_queries": total_queries,
        }

        # Add RAGAS metrics if computed
        if use_ragas and ragas_totals:
            for metric_name, total in ragas_totals.items():
                results[k][f"avg_{metric_name}"] = total / total_queries
    results["estimated_memory_usage_MB"] = (
        vectorstore.index.ntotal * vectorstore.index.d * 4 / (1024 * 1024)
    )
    return results


def save_results(results: dict, output_file: Path):
    """
    Save the results to a JSON file.
    """
    with open(output_file, "w") as f:
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
    with open(results_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    results = load_results(Path("data/FAISS_evaluation_results.json"))
    graph_results(results, Path("data/FAISS_evaluation_results.png"))
    exit()
