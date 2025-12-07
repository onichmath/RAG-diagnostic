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
from src.ingest.langchain_ingest import LangChainIngest, list_available_datasets
from src.retriever.faiss_builder import (
    build_faiss_index,
    build_faiss_index_fast,
    load_faiss_index,
)
from src.eval.eval import evaluate_rag_system, save_results, graph_results
from src.reader.reader import Generator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build RAG pipeline using LangChain (simplified)"
    )

    # Data loading options
    parser.add_argument(
        "--pubmed-docs",
        type=int,
        default=100000,
        help="Number of PubMed documents to load",
    )
    parser.add_argument(
        "--textbook-docs",
        type=int,
        default=50000,
        help="Number of textbook documents to load",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force download even if local copies exist (default: False)",
    )

    # Processing options
    parser.add_argument(
        "--chunk-size", type=int, default=300, help="Chunk size for text processing"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=60,
        help="Chunk overlap for text processing",
    )
    parser.add_argument(
        "--separators",
        type=list,
        default=["\n\n", "\n", " ", ""],
        help="Separators for text processing",
    )
    parser.add_argument(
        "--text-splitter",
        type=str,
        default="RecursiveCharacterTextSplitter",
        help="Text splitter to use",
    )

    # Model options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="thenlper/gte-small",
        help="HuggingFace embedding model to use",
    )

    # FAISS optimization options
    parser.add_argument(
        "--use-fast-faiss",
        action="store_true",
        default=False,
        help="Use optimized FAISS index building (default: True)",
    )
    parser.add_argument(
        "--faiss-batch-size",
        type=int,
        default=2000,
        help="Batch size for FAISS index building",
    )
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        default="IVF",
        choices=["IVF", "HNSW", "Flat"],
        help="Type of FAISS index to build",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU for embeddings (default: True)",
    )

    # Test options
    parser.add_argument(
        "--test-queries-file",
        type=str,
        default="data/requests/queries.json",
        help="Path to JSON file containing test queries",
    )
    parser.add_argument(
        "--max-test-queries",
        type=int,
        default=10,
        help="Maximum number of test queries to run",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,
        help="Skip retrieval testing (default: False)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/FAISS_evaluation_results.json",
        help="Path to JSON file containing evaluation results",
    )
    parser.add_argument(
        "--graph-file",
        type=str,
        default="data/FAISS_evaluation_results.png",
        help="Path to PNG file containing evaluation results graph",
    )
    parser.add_argument(
        "--k-array",
        type=list,
        default=[5, 10, 20, 50, 100],
        help="Array of k values to evaluate",
    )
    parser.add_argument(
        "--save-results",
        action="store_false",
        default=True,
        help="Save evaluation results to a JSON file (default: True)",
    )

    # Filter options
    parser.add_argument(
        "--filter-pubmed",
        action="store_true",
        default=False,
        help="Apply heuristic filtering on PubMed chunks before indexing",
    )

    parser.add_argument(
        "--use-llm-reranker",
        action="store_true",
        default=False,
        help="Use local LLM title-based reranker",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="local",
        help="Ignored for local reranking; kept for compatibility",
    )

    # colber rerank options
    parser.add_argument(
        "--use-colbert-reranker",
        action="store_true",
        default=False,
        help="Use ColBERT-style Late Interaction reranker",
    )

    parser.add_argument(
        "--colbert-model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model for ColBERT reranker",
    )

    # RAGAS evaluation options
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        default=False,
        help="Compute RAGAS metrics (requires generator)",
    )

    parser.add_argument(
        "--ragas-model",
        type=str,
        default="local",
        help="Model name for RAGAS judge (default: 'local' uses microsoft/phi-2)",
    )

    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        help="Model name for answer generation (default: phi for Ollama, microsoft/phi-2 for HF)",
    )

    parser.add_argument(
        "--use-ollama",
        action="store_true",
        default=False,
        help="Use Ollama for generator (default: False, uses shared model from llm_title_rerank)",
    )
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
    logger.info(
        f"Using document limits: PubMed={args.pubmed_docs:,}, Textbooks={args.textbook_docs:,}"
    )

    ingest = LangChainIngest(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=args.separators,
    )

    loaded = ingest.load_all_medrag_data_to_disk(
        corpus_dir=Path("data/corpus_raw"),
        pubmed_docs=args.pubmed_docs,
        textbook_docs=args.textbook_docs,
        force_download=args.force_download,
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
    # Step 4: Build FAISS index if not exists
    index_path = Path("data/indices/faiss_index")
    if not index_path.exists():
        logger.info("Step 4: Building FAISS index...")
        logger.info(f"Using embedding model: {args.embedding_model}")
        logger.info(f"Using fast FAISS: {args.use_fast_faiss}")
        logger.info(f"Filter PubMed: {args.filter_pubmed}")

        if args.use_fast_faiss:
            index_path = build_faiss_index_fast(
                corpus_dir=Path("data/corpus_raw"),
                # corpus_norm_dir=guidelines_path,
                corpus_norm_dir=Path("data/corpus_norm"),
                output_dir=index_path,
                embedding_model=args.embedding_model,
                batch_size=args.faiss_batch_size,
                use_gpu=args.use_gpu,
                faiss_index_type=args.faiss_index_type,
                filter_pubmed=args.filter_pubmed,
            )
        else:
            index_path = build_faiss_index(
                corpus_dir=Path("data/corpus_raw"),
                # corpus_norm_dir=guidelines_path,
                corpus_norm_dir=Path("data/corpus_norm"),
                output_dir=index_path,
                embedding_model=args.embedding_model,
                batch_size=args.faiss_batch_size,
                use_gpu=args.use_gpu,
                filter_pubmed=args.filter_pubmed,
            )

        logger.info(f"FAISS index built and saved to: {index_path}")
    else:
        logger.info(f"FAISS index already exists at: {index_path}")

    # Step 5: Test retrieval (optional)
    if not args.skip_test:
        logger.info("Step 5: Testing retrieval...")

        generator = None
        if args.use_ragas:
            logger.info("Initializing Generator for RAGAS evaluation...")
            logger.info(f"Generator model: {args.generator_model or 'default'}")
            logger.info(f"Using Ollama: {args.use_ollama}")
            try:
                if args.use_ollama:
                    generator = Generator(
                        model_name=args.generator_model, use_ollama=True
                    )
                else:
                    from src.reranker.llm_title_rerank import (
                        get_shared_model,
                        get_shared_tokenizer,
                    )

                    logger.info("Reusing shared model from llm_title_rerank")
                    generator = Generator(
                        model_name=args.generator_model,
                        use_ollama=False,
                        shared_model=get_shared_model(),
                        shared_tokenizer=get_shared_tokenizer(),
                    )
                logger.info("Generator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Generator: {e}")
                logger.error("RAGAS evaluation requires a working Generator. Exiting.")
                exit(1)

        results = evaluate_rag_system(
            index_path=index_path,
            queries_file=args.test_queries_file,
            max_queries=args.max_test_queries,
            k_array=args.k_array,
            use_llm_reranker=args.use_llm_reranker,
            llm_model=args.llm_model,
            use_colbert_reranker=args.use_colbert_reranker,
            colbert_model=args.colbert_model,
            generator=generator,
            use_ragas=args.use_ragas,
            ragas_model=args.ragas_model,
        )

        if args.save_results:
            save_results(results, Path(args.output_file))
            # graph_results(results, Path(args.graph_file))

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
