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
    load_medrag_data_simple,
    process_guidelines_simple,
    list_available_datasets_simple
)
from src.retriever.faiss_builder import build_faiss_index, build_faiss_index_fast, load_faiss_index

# Configure logging
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
    
    logger.info("🚀 Starting RAG Diagnostic Pipeline (LangChain)")
    logger.info("=" * 60)
    
    # Step 1: Check existing datasets
    logger.info("Step 1: Checking existing datasets...")
    existing_datasets = list_available_datasets_simple(Path("data/corpus_raw"))
    if existing_datasets:
        logger.info("Existing datasets:")
        for name, count in existing_datasets.items():
            logger.info(f"  📂 {name}: {count:,} documents")
    else:
        logger.info("No existing datasets found")
    
    # Step 2: Load MedRAG datasets using LangChain
    logger.info("Step 2: Loading MedRAG datasets using LangChain...")
    logger.info(f"Using document limits: PubMed={args.pubmed_docs:,}, Textbooks={args.textbook_docs:,}")
    
    loaded = load_medrag_data_simple(
        corpus_dir=Path("data/corpus_raw"),
        pubmed_docs=args.pubmed_docs,
        textbook_docs=args.textbook_docs,
        force_download=args.force_download
    )
    
    logger.info("Loaded datasets:")
    for name, path in loaded.items():
        logger.info(f"  📁 {name}: {path}")
    
    # Step 3: Process guidelines using LangChain
    logger.info("Step 3: Processing guidelines using LangChain...")
    guidelines_path = process_guidelines_simple(
        guidelines_dir=Path("data/guidelines"),
        output_dir=Path("data/corpus_norm"),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    logger.info(f"Guidelines processed and saved to: {guidelines_path}")
    
    # Step 4: Build FAISS index
    logger.info("Step 4: Building FAISS index...")
    logger.info(f"Using embedding model: {args.embedding_model}")
    
    if args.use_fast_faiss:
        logger.info("Using optimized FAISS index building...")
        logger.info(f"FAISS settings: batch_size={args.faiss_batch_size}, index_type={args.faiss_index_type}, use_gpu={args.use_gpu}")
        index_path = build_faiss_index_fast(
            corpus_dir=Path("data/corpus_raw"),
            corpus_norm_dir=Path("data/corpus_norm"),
            output_dir=Path("data/indices"),
            embedding_model=args.embedding_model,
            batch_size=args.faiss_batch_size,
            use_gpu=args.use_gpu,
            faiss_index_type=args.faiss_index_type
        )
    else:
        logger.info("Using standard FAISS index building...")
        index_path = build_faiss_index(
            corpus_dir=Path("data/corpus_raw"),
            corpus_norm_dir=Path("data/corpus_norm"),
            output_dir=Path("data/indices"),
            embedding_model=args.embedding_model,
            batch_size=args.faiss_batch_size,
            use_gpu=args.use_gpu
        )
    
    logger.info(f"FAISS index built and saved to: {index_path}")
    
    # Step 5: Test retrieval (optional)
    if not args.skip_test:
        logger.info("Step 5: Testing retrieval...")
        
        # Load test queries
        queries_file = Path(args.test_queries_file)
        test_queries = load_test_queries(queries_file, args.max_test_queries)
        
        if not test_queries:
            logger.warning("No test queries available, skipping retrieval testing")
        else:
            vectorstore = load_faiss_index(index_path)
            
            for i, query_data in enumerate(test_queries, 1):
                query_id = query_data.get('query_id', f'q{i:03d}')
                query_text = query_data.get('query_text', '')
                expected_docs = query_data.get('expected_gold_docs', [])
                
                logger.info(f"\nTest Query {i} ({query_id}): {query_text}")
                logger.info(f"Expected documents: {expected_docs}")
                
                results = vectorstore.similarity_search(query_text, k=3)
                
                logger.info("Top 3 results:")
                for j, doc in enumerate(results, 1):
                    doc_id = doc.metadata.get('doc_id', 'Unknown')
                    source = doc.metadata.get('source', 'Unknown')
                    is_expected = doc_id in expected_docs if expected_docs else False
                    expected_marker = " ✅" if is_expected else ""
                    
                    logger.info(f"  {j}. {doc.page_content[:150]}...")
                    logger.info(f"     Source: {source}")
                    logger.info(f"     Doc ID: {doc_id}{expected_marker}")
                
                # Check if any expected documents were found
                found_expected = any(doc.metadata.get('doc_id') in expected_docs for doc in results)
                if expected_docs:
                    if found_expected:
                        logger.info("  ✅ Found expected document(s)")
                    else:
                        logger.info("  ❌ No expected documents found in top 3 results")
    else:
        logger.info("Step 5: Skipping retrieval testing")
    
    logger.info("\n✅ LangChain pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()