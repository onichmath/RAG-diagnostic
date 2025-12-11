# MED RAG - Diagnostic Retrieval System

A healthcare domain-specific RAG (Retrieval-Augmented Generation) system for evaluating and improving retrieval performance on medical queries using clinical guidelines, PubMed articles, and medical textbooks.

## Project Overview

Modern healthcare requires fast, accurate access to evidence-based information. This project implements a comprehensive RAG pipeline tailored for biomedical information seeking, featuring:

- **Multi-source Corpus**: Clinical guidelines, PubMed abstracts, and medical textbooks
- **Advanced Retrieval**: FAISS vector stores with multiple index types (Flat, IVF, HNSW)
- **Query Transformation**: LLM-powered query rewriting, expansion, decomposition, and step-back prompting
- **Reranking**: ColBERT late-interaction reranker and LLM-based title reranker
- **Comprehensive Evaluation**: Precision@k, NDCG@k, Context Precision/Recall, throughput, and latency metrics
- **RAGAS Integration**: LLM-as-a-judge evaluation for contextual generation quality

## Key Features

### Phase 1: Baseline Retrieval
- FAISS FlatL2 index for exact nearest-neighbor search
- GTE-small embedding model (384-dim)
- Baseline metrics: Precision@k, NDCG@k, latency, throughput

### Phase 2: Advanced Retrieval
- **Dataset Filtering**: Remove non-clinical PubMed entries (~40% reduction) for improved domain relevance
- **Query Transformation**: Gemini-powered query rewriting, expansion, and decomposition
- **ColBERT Reranker**: Token-level late-interaction reranking for fine-grained clinical matching
- **LLM Title Reranker**: Local Phi-2 based title reranking
- **RAGAS Metrics**: Context precision and recall using GPT-4o-mini as judge

## Project Structure

```
RAG-diagnostic/
├── data/                       # All data artifacts
│   ├── corpus_raw/             # Raw datasets (MedRAG PubMed, Textbooks)
│   ├── corpus_norm/            # Processed datasets (chunked guidelines)
│   ├── guidelines/             # Clinical guideline PDFs (not in repo)
│   ├── requests/               # Test queries and evaluation data
│   │   └── queries.json        # Test query set with golden answers
│   ├── indices/                # FAISS vector indices
│   └── test_results/           # Evaluation output files
├── src/                        # Source code
│   ├── ingest/                 # Data downloading and preprocessing
│   │   └── langchain_ingest.py # LangChain-based document processing
│   ├── retriever/              # Vector store and retrieval
│   │   └── faiss_builder.py    # FAISS index construction
│   ├── reranker/               # Reranking modules
│   │   ├── colbert_rerank.py   # ColBERT late-interaction reranker
│   │   └── llm_title_rerank.py # LLM-based title reranker
│   ├── query/                  # Query processing
│   │   └── query_transformer.py # Query transformation module
│   ├── reader/                 # Answer generation
│   │   └── reader.py           # RAG generator
│   └── eval/                   # Evaluation
│       ├── eval.py             # Main evaluation pipeline
│       └── llm_metrics.py      # RAGAS metrics computation
├── scripts/                    # Executable scripts
│   └── build_pipeline.py       # Main pipeline script
├── notebooks/                  # Jupyter notebooks for analysis
├── env.yml                     # Conda environment file
└── requirements.txt            # Pip requirements
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone git@github.com:onichmath/RAG-diagnostic.git
cd RAG-diagnostic

# Option A: Using Conda (recommended)
conda env create -f env.yml
conda activate rag-diagnostic

# Option B: Using pip
pip install -r requirements.txt
```

### 2. Download Clinical Guidelines

**Important**: Clinical guideline PDFs are not included in the repository due to copyright restrictions. Download them separately and place in `data/guidelines/`:

| Guideline | Source | Save As |
|-----------|--------|---------|
| Surviving Sepsis Campaign 2021 | [Springer](https://link.springer.com/article/10.1007/s00134-021-06506-y) | `surviving_sepsis.pdf` |
| ADA Standards of Care in Diabetes 2024 | [Diabetes Journals](https://diabetesjournals.org/care/article/47/Supplement_1/S1/153952/Introduction-and-Methodology-Standards-of-Care-in) | `ada_soc_diabetes_2024.pdf` |
| ACC/AHA Heart Failure Guideline 2022 | [AHA Journals](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063) | `acc_aha_hf.pdf` |
| ACC/AHA/HRS Atrial Fibrillation 2019 | [AHA Journals](https://www.ahajournals.org/doi/10.1161/CIR.0000000000000665) | `acc_aha_afib.pdf` |
| AHA Stroke Prevention 2021 | [AHA Journals](https://www.ahajournals.org/doi/10.1161/STR.0000000000000375) | `aha_stroke_2021.pdf` |
| IDSA COVID-19 Guidelines 2025 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40831386/) | `idsa_clinical_guidelines_covid19.pdf` |

### 3. Run the Pipeline

```bash
# Run complete pipeline with default settings
python scripts/build_pipeline.py

# Run with custom parameters
python scripts/build_pipeline.py \
    --pubmed-docs 100000 \
    --textbook-docs 50000 \
    --max-test-queries 10
```

## Command-Line Options

### Data Loading Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pubmed-docs` | 100000 | Number of PubMed documents to load |
| `--textbook-docs` | 50000 | Number of textbook documents to load |
| `--force-download` | False | Force re-download even if local copies exist |
| `--filter-pubmed` | False | Apply heuristic filtering to remove non-clinical PubMed entries |

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk-size` | 300 | Chunk size for text processing |
| `--chunk-overlap` | 60 | Chunk overlap for text processing |
| `--embedding-model` | `thenlper/gte-small` | HuggingFace embedding model |

### FAISS Index Options

| Option | Default | Description |
|--------|---------|-------------|
| `--faiss-index-type` | `IVF` | Index type: `Flat`, `IVF`, or `HNSW` |
| `--faiss-batch-size` | 2000 | Batch size for index building |
| `--use-fast-faiss` | False | Use optimized FAISS index building |
| `--use-gpu` | True | Use GPU for embeddings |
| `--rebuild-index` | False | Force rebuild of FAISS index |

### Reranking Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use-colbert-reranker` | False | Enable ColBERT late-interaction reranker |
| `--colbert-model` | `bert-base-uncased` | Model for ColBERT reranker |
| `--use-llm-reranker` | False | Enable LLM-based title reranker |

### Query Transformation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--query-transform-model` | `gemini-2.5-flash` | Model for query transformation |
| `--query-transform-provider` | `auto` | Provider: `auto`, `colab`, or `gemini` |
| `--gemini-api-key` | None | Gemini API key (or set `GEMINI_API_KEY` env var) |

### RAGAS Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use-ragas` | False | Enable RAGAS metrics computation |
| `--ragas-model` | `gpt-4o-mini` | Model for RAGAS judge |
| `--generator-model` | None | Model for answer generation |
| `--use-ollama` | False | Use Ollama for generator |

### Test & Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-queries-file` | `data/requests/queries.json` | Path to test queries file |
| `--max-test-queries` | 10 | Maximum number of test queries |
| `--skip-test` | False | Skip retrieval testing |
| `--k-array` | `[5, 10, 20]` | k values for evaluation |
| `--output-file` | `data/FAISS_evaluation_results.json` | Output file for results |
| `--save-results` | True | Save evaluation results |

## Example Configurations

### Baseline Retrieval (Fast)
```bash
python scripts/build_pipeline.py \
    --max-test-queries 10 \
    --k-array [5,10,20]
```

### Filtered Dataset with ColBERT Reranking
```bash
python scripts/build_pipeline.py \
    --filter-pubmed \
    --use-colbert-reranker \
    --max-test-queries 10
```

### Full Pipeline with Query Transformation
```bash
export GEMINI_API_KEY="your-api-key"
python scripts/build_pipeline.py \
    --filter-pubmed \
    --use-colbert-reranker \
    --query-transform-provider gemini \
    --query-transform-model gemini-2.5-flash
```

### RAGAS Evaluation with OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
python scripts/build_pipeline.py \
    --use-ragas \
    --ragas-model gpt-4o-mini \
    --use-colbert-reranker
```

## Evaluation Metrics

### Traditional IR Metrics
- **Precision@k**: Fraction of retrieved documents that are relevant
- **NDCG@k**: Normalized Discounted Cumulative Gain for ranking quality
- **Latency**: Query response time in milliseconds
- **Throughput**: Queries per second (QPS)

### RAGAS Metrics (LLM-as-Judge)
- **Context Precision**: How well the generated answer is supported by retrieved passages
- **Context Recall**: How much of the reference answer is recoverable from retrieved contexts

## Results Summary

| Configuration | Precision@10 | NDCG@10 | Latency (ms) | Notes |
|--------------|--------------|---------|--------------|-------|
| Baseline | 0.60 | 0.65 | 25.9 | FlatL2 index |
| + PubMed Filter | 0.60 | 0.65 | 22.6 | 25% faster, 25% less memory |
| + Query Transform | 0.62 | 0.67 | 35.2 | +3.3% precision |
| + ColBERT Reranker | 0.60 | 0.67 | 102 | Best NDCG |
| + LLM Reranker | 0.60 | 0.65 | 2921 | High latency |

## System Requirements

- **Hardware**: AMD EPYC 7D12 or equivalent, NVIDIA RTX 3090 (recommended for GPU acceleration)
- **Memory**: ~234MB for baseline index, ~174MB with filtering
- **Python**: 3.10+

## Team

- Deep Saran Masanam
- Matthew Omalley-Nichols
- Marc Fehlhaber
- Suprith Krishnakumar
- Shreeyash Pacharne

## License

This project is open-sourced under the MIT License.

## Citations

1. [MedRAG Dataset](https://huggingface.co/MedRAG)
2. [PubMedQA Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA)
3. [FAISS Library](https://github.com/facebookresearch/faiss)
