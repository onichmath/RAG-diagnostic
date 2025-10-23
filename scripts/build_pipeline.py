#!/usr/bin/env python3
"""
End-to-end pipeline to build FAISS vector store from MedRAG datasets and guidelines.
Uses LangChain for simplified document processing.
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.ingest.langchain_ingest import (
    LangChainIngest,
    load_medrag_data,
    process_guidelines,
    list_available_datasets
)
from src.retriever.faiss_builder import build_faiss_index, build_faiss_index_fast, load_faiss_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build RAG pipeline using LangChain (simplified)")
    
    # Data loading options
    parser.add_argument("--pubmed-docs", type=int, default=100000,
                       help="Number of PubMed documents to load")
    parser.add_argument("--textbook-docs", type=int, default=50000,
                       help="Number of textbook documents to load")
    parser.add_argument("--force-download", action="store_true",
                       help="Force download even if local copies exist")
    
    # Processing options
    parser.add_argument("--chunk-size", type=int, default=300,
                       help="Chunk size for text processing")
    parser.add_argument("--chunk-overlap", type=int, default=60,
                       help="Chunk overlap for text processing")
    
    # Model options
    parser.add_argument("--embedding-model", type=str, default="thenlper/gte-small",
                       help="HuggingFace embedding model to use")
    
    # FAISS optimization options
    parser.add_argument("--use-fast-faiss", action="store_true",
                       help="Use optimized FAISS index building")
    parser.add_argument("--faiss-batch-size", type=int, default=2000,
                       help="Batch size for FAISS index building")
    parser.add_argument("--faiss-index-type", type=str, default="IVF",
                       choices=["IVF", "HNSW", "Flat"],
                       help="Type of FAISS index to build")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                       help="Use GPU for embeddings (default: True)")
    
    # Test options
    parser.add_argument("--test-queries-file", type=str, 
                       default="data/requests/queries.json",
                       help="Path to JSON file containing test queries")
    parser.add_argument("--max-test-queries", type=int, default=5,
                       help="Maximum number of test queries to run")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip retrieval testing")
    
    return parser.parse_args()


def load_test_queries(queries_file: Path, max_queries: int = 5) -> list:
    """Load test queries from JSON file."""
    try:
        with open(queries_file, 'r') as f:
            data = json.load(f)
        
        queries = data.get('queries', [])
        queries = queries[:max_queries]
        
        logger.info(f"Loaded {len(queries)} test queries from {queries_file}")
        return queries
        
    except Exception as e:
        logger.error(f"Error loading test queries: {e}")
        return []


def main():
    """Run the complete pipeline using LangChain."""
    args = parse_args()
    
    logger.info("Starting RAG Diagnostic Pipeline (LangChain)")
    logger.info("=" * 60)
    
    # Step 1: Check existing datasets
    logger.info("Step 1: Checking existing datasets...")
    existing_datasets = list_available_datasets(Path("data/corpus_raw"))
    if existing_datasets:
        logger.info("Existing datasets:")
        for name, count in existing_datasets.items():
            logger.info(f"  📂 {name}: {count:,} documents")
    else:
        logger.info("No existing datasets found")
    
    # Step 2: Load MedRAG datasets using LangChain
    logger.info("Step 2: Loading MedRAG datasets using LangChain...")
    logger.info(f"Using document limits: PubMed={args.pubmed_docs:,}, Textbooks={args.textbook_docs:,}")
    
    loaded = load_medrag_data(
        corpus_dir=Path("data/corpus_raw"),
        pubmed_docs=args.pubmed_docs,
        textbook_docs=args.textbook_docs,
        force_download=args.force_download
    )
    
    logger.info("Loaded datasets:")
    for name, path in loaded.items():
        logger.info(f"{name}: {path}")
    
    # Step 3: Process guidelines using LangChain if not exists
    guidelines_path = Path("data/corpus_norm/guidelines_processed")
    if not guidelines_path.exists():
        logger.info("Step 3: Processing guidelines using LangChain...")
        guidelines_path = process_guidelines(
            guidelines_dir=Path("data/guidelines"),
            output_dir=Path("data/corpus_norm"),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        logger.info(f"Guidelines processed and saved to: {guidelines_path}")
    else:
        logger.info(f"Guidelines already processed and saved to: {guidelines_path}")
    
    # Step 4: Build FAISS index if not exists
    index_path = Path("data/indices/faiss_index")
    if not index_path.exists():
        logger.info("Step 4: Building FAISS index...")
        logger.info(f"Using embedding model: {args.embedding_model}")
        index_path = build_faiss_index(
            corpus_dir=Path("data/corpus_raw"),
            corpus_norm_dir=guidelines_path,
            output_dir=index_path,
            embedding_model=args.embedding_model,
            batch_size=args.faiss_batch_size,
            use_gpu=args.use_gpu
        )
        logger.info(f"FAISS index built and saved to: {index_path}")
    else:
        logger.info(f"FAISS index already exists at: {index_path}")
    
    
    # Step 5: Test retrieval (optional)
    if not args.skip_test:
        logger.info("Step 5: Testing retrieval...")
        
        queries_file = Path(args.test_queries_file)
        test_queries = load_test_queries(queries_file, args.max_test_queries)
        vectorstore = load_faiss_index(index_path) 
        
        total_precision_at_k = 0.0
        total_queries = len(test_queries)
        
        logger.info(f"Running evaluation on {total_queries} queries...")
        k = 50
        
        for i, query_data in enumerate(test_queries, 1):
            query_id = query_data.get('query_id', f'q{i:03d}')
            query_text = query_data.get('query_text', '')
            expected_docs = query_data.get('expected_gold_docs', [])
            

            results = vectorstore.similarity_search(query_text, k=k)
            
            retrieved_titles = []
            retrieved_doc_ids = []
            
            for j, doc in enumerate(results, 1):
                title = doc.metadata.get('title', 'Unknown')
                doc_id = doc.metadata.get('doc_id', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                
                retrieved_titles.append(title)
                retrieved_doc_ids.append(doc_id)

            expected_doc_patterns = []
            for expected_doc in expected_docs:
                if expected_doc == "idsa_clinical_guideline_covid19":
                    expected_doc_patterns.extend(["idsa", "covid", "covid19", "clinical_guideline"])
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
            
            relevant_retrieved = []
            for j, (title, doc_id) in enumerate(zip(retrieved_titles, retrieved_doc_ids)):
                title_lower = title.lower()
                doc_id_lower = doc_id.lower()
                
                for pattern in expected_doc_patterns:
                    if pattern.lower() in title_lower or pattern.lower() in doc_id_lower:
                        relevant_retrieved.append(j)
                        break
            
            num_relevant_retrieved = len(relevant_retrieved)
            num_expected = len(expected_docs)
            
            precision_at_k = num_relevant_retrieved / k if k > 0 else 0.0
            
            total_precision_at_k += precision_at_k
            
            logger.info(f"Relevant docs found: {num_relevant_retrieved}/{num_expected}")
            logger.info(f"Precision@{k}: {precision_at_k:.3f}")
            
        
        avg_precision_at_k = total_precision_at_k / total_queries
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Total queries evaluated: {total_queries}")
        logger.info(f"Average Precision@{k}: {avg_precision_at_k:.3f}")
        logger.info("="*60)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()