#!/usr/bin/env python3
"""
End-to-end pipeline to build FAISS vector store from MedRAG datasets and guidelines.
Uses LangChain for simplified document processing.
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.ingest.langchain_ingest import (
    LangChainIngest,
    list_available_datasets
)
from src.retriever.faiss_builder import build_faiss_index, build_faiss_index_fast, load_faiss_index
from src.eval.eval import evaluate_rag_system, save_results

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
    parser.add_argument("--separators", type=list, default=["\n\n", "\n", " ", ""],
                       help="Separators for text processing")
    parser.add_argument("--text-splitter", type=str, default="RecursiveCharacterTextSplitter",
                       help="Text splitter to use")
    
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
    parser.add_argument("--max-test-queries", type=int, default=10,
                       help="Maximum number of test queries to run")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip retrieval testing")
    parser.add_argument("--output-file", type=str, default="data/FAISS_evaluation_results.json",
                       help="Path to JSON file containing evaluation results") 
    parser.add_argument("--k-array", type=list, default=[5, 10, 20, 50, 100],
                       help="Array of k values to evaluate")
    
    return parser.parse_args()




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
            logger.info(f"{name}: {count:,} documents")
    else:
        logger.info("No existing datasets found")
    
    # Step 2: Load MedRAG datasets using LangChain
    logger.info("Step 2: Loading MedRAG datasets using LangChain...")
    logger.info(f"Using document limits: PubMed={args.pubmed_docs:,}, Textbooks={args.textbook_docs:,}")

    ingest = LangChainIngest(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=args.separators, 
    )
    
    loaded = ingest.load_all_medrag_data_to_disk(
        corpus_dir=Path("data/corpus_raw"),
        pubmed_docs=args.pubmed_docs,
        textbook_docs=args.textbook_docs,
        force_download=args.force_download
    )
    
    logger.info("Loaded datasets:")
    for name, path in loaded.items():
        logger.info(f"{name}: {path}")
    
    # Step 3: Process guidelines using LangChain if not processed
    if len(ingest.load_pdfs_from_directory(Path("data/guidelines"))) < 6:
        logger.error("Please download the guidelines to data/guidelines folder")
        exit(1)
    
    logger.info("Step 3: Processing guidelines using LangChain...")
    guidelines_path = Path("data/corpus_norm/guidelines_processed/")
    if not guidelines_path.exists():
        guidelines_path = ingest.process_guidelines_to_disk(
            guidelines_dir=Path("data/guidelines"),
            output_dir=Path("data/corpus_norm"),
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
        k_array = args.k_array
        results = evaluate_rag_system(index_path, args.test_queries_file, args.max_test_queries, k_array) 
        save_results(results, Path(args.output_file))
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()