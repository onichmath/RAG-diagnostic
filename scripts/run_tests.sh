#!/bin/bash
# Test script for RAG pipeline with different configurations
# Saves results to separate files for comparison
#
# Usage:
#   ./scripts/run_tests.sh                    # Run all tests
#   ./scripts/run_tests.sh --skip-index      # Skip index building (use existing)
#   ./scripts/run_tests.sh --rebuild-index   # Force rebuild index

set -e  # Exit on error

# Env:
# CPU: AMD EPYC 7D12 32-Core Processor
# GPU: RTX 3090, 24GB VRAM
# Model: meta-llama/Llama-3.2-3B-Instruct
# Embedding Model: thenlper/gte-small



# Configuration
# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Normalize path to remove any trailing slashes
BASE_DIR="${BASE_DIR%/}"

# Validate BASE_DIR
if [ -z "$BASE_DIR" ] || [ "$BASE_DIR" = "/" ]; then
    echo "Error: Invalid BASE_DIR: $BASE_DIR" >&2
    echo "Script location: ${BASH_SOURCE[0]}" >&2
    exit 1
fi

RESULTS_DIR="$(echo "${BASE_DIR}/data/test_results" | sed 's|//|/|g')"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "RAG Pipeline Test Suite (RAGAS Only)"
echo "Results will be saved to: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

PUBMED_DOCS=100000
TEXTBOOK_DOCS=50000
MAX_TEST_QUERIES=10
EMBEDDING_MODEL="thenlper/gte-small"  # Change this to use a different embedding model for both FAISS and RAGAS

# Test 1: RAGAS only (no rerankers)
echo "[TEST 1/8] RAGAS Metrics (no rerankers)"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-ragas \
    --ragas-model "local" \
    --output-file "${RESULTS_DIR}/ragas_only_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu 
    #--rebuild-index \
echo "✓ Saved to: ragas_only_${TIMESTAMP}.json"
echo ""

# Test 2: LLM Reranker + RAGAS
echo "[TEST 2/8] LLM Reranker + RAGAS"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-llm-reranker \
    --use-ragas \
    --ragas-model "local" \
    --output-file "${RESULTS_DIR}/llm_reranker_ragas_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu
echo "✓ Saved to: llm_reranker_ragas_${TIMESTAMP}.json"
echo ""

# Test 3: ColBERT Reranker + RAGAS
echo "[TEST 3/8] ColBERT Reranker + RAGAS"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-colbert-reranker \
    --colbert-model "bert-base-uncased" \
    --use-ragas \
    --ragas-model "local" \
    --output-file "${RESULTS_DIR}/colbert_reranker_ragas_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu
echo "✓ Saved to: colbert_reranker_ragas_${TIMESTAMP}.json"
echo ""

# Test 4: Full pipeline with all features
echo "[TEST 4/8] Full Pipeline: LLM Reranker + ColBERT + RAGAS"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-llm-reranker \
    --use-colbert-reranker \
    --colbert-model "bert-base-uncased" \
    --use-ragas \
    --ragas-model "local" \
    --output-file "${RESULTS_DIR}/full_pipeline_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu
echo "✓ Saved to: full_pipeline_${TIMESTAMP}.json"
echo ""

echo "=========================================="
echo "Starting filtered PubMed dataset tests"
echo "=========================================="
echo ""

# Test 5: RAGAS only with filtered PubMed (no rerankers)
echo "[TEST 5/8] RAGAS Metrics (filtered PubMed, no rerankers)"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-ragas \
    --ragas-model "local" \
    --filter-pubmed \
    --output-file "${RESULTS_DIR}/filtered_ragas_only_${TIMESTAMP}.json" \
    --save-results \
    --rebuild-index \
    --use-gpu
echo "✓ Saved to: filtered_ragas_only_${TIMESTAMP}.json"
echo ""

# Test 6: LLM Reranker + RAGAS with filtered PubMed
echo "[TEST 6/8] LLM Reranker + RAGAS (filtered PubMed)"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-llm-reranker \
    --use-ragas \
    --ragas-model "local" \
    --filter-pubmed  \
    --output-file "${RESULTS_DIR}/filtered_llm_reranker_ragas_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu
echo "✓ Saved to: filtered_llm_reranker_ragas_${TIMESTAMP}.json"
echo ""

# Test 7: ColBERT Reranker + RAGAS with filtered PubMed
echo "[TEST 7/8] ColBERT Reranker + RAGAS (filtered PubMed)"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-colbert-reranker \
    --colbert-model "bert-base-uncased" \
    --use-ragas \
    --ragas-model "local" \
    --filter-pubmed  \
    --output-file "${RESULTS_DIR}/filtered_colbert_reranker_ragas_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu
echo "✓ Saved to: filtered_colbert_reranker_ragas_${TIMESTAMP}.json"
echo ""

# Test 8: Full pipeline with all features and filtered PubMed
echo "[TEST 8/8] Full Pipeline: LLM Reranker + ColBERT + RAGAS (filtered PubMed)"
python scripts/build_pipeline.py \
    --pubmed-docs $PUBMED_DOCS \
    --textbook-docs $TEXTBOOK_DOCS \
    --max-test-queries $MAX_TEST_QUERIES \
    --embedding-model "$EMBEDDING_MODEL" \
    --use-llm-reranker \
    --use-colbert-reranker \
    --colbert-model "bert-base-uncased" \
    --use-ragas \
    --ragas-model "local" \
    --filter-pubmed  \
    --output-file "${RESULTS_DIR}/filtered_full_pipeline_${TIMESTAMP}.json" \
    --save-results \
    --use-gpu
echo "✓ Saved to: filtered_full_pipeline_${TIMESTAMP}.json"
echo ""








echo "=========================================="
echo "All tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="
echo ""
echo "Summary of test files:"
ls -lh "${RESULTS_DIR}"/*"${TIMESTAMP}"*.json 2>/dev/null || echo "No results files found"
echo ""
echo "To compare results, run:"
echo "  python scripts/compare_results.py ${RESULTS_DIR}/*${TIMESTAMP}*.json"
echo ""
echo "Or view individual results:"
echo "  cat ${RESULTS_DIR}/ragas_only_${TIMESTAMP}.json | python -m json.tool"

