# RAG Diagnostic Pipeline

A healthcare domain-specific RAG (Retrieval-Augmented Generation) system for evaluating and improving retrieval performance on medical queries using clinical guidelines, PubMed articles, and medical textbooks.

## 🎯 Project Overview

This project implements a comprehensive RAG system specifically designed for medical information retrieval, featuring:

- **Multi-source Data**: Clinical guidelines, PubMed articles, medical textbooks
- **Advanced Retrieval**: FAISS vector stores with multiple index types
- **Evaluation**: Precision@k metrics
- **Medical Domain Focus**: Optimized for healthcare queries and terminology

## 📁 Project Structure

```
RAG-diagnostic/
├── data/                    # All data artifacts
│   ├── corpus_raw/         # Raw datasets (MedRAG, guidelines)
│   ├── corpus_norm/         # Processed datasets (chunked)
│   ├── indices/            # Vector indices (FAISS)
│   ├── guidelines/         # Clinical guideline PDFs (not in repo)
│   └── requests/           # Test queries and evaluation data
├── src/                    # Source code
│   ├── ingest/            # Data downloading and preprocessing
│   └── retriever/         # Vector store and retrieval
├── scripts/               # Executable scripts
│   └── build_pipeline.py  # Main pipeline script
├── notebooks/             # Jupyter notebooks for analysis
├── config/                # Configuration files
├── infra/                 # Infrastructure as code
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG-diagnostic

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Clinical Guidelines

**⚠️ Important**: Clinical guideline PDFs are not included in the repository due to copyright restrictions. You need to download them separately:

```bash
# Download clinical guidelines to data/guidelines/
# Required files:
# - acc_aha_hf.pdf (Heart Failure Guidelines)
# - ada_soc_diabetes_2024.pdf (Diabetes Guidelines)
# - aha_acc_afib.pdf (Atrial Fibrillation Guidelines)
# - aha_stroke_2021.pdf (Stroke Guidelines)
# - idsa_clinical_guideline_covid19.pdf (COVID-19 Guidelines)
# - surviving_sepsis.pdf (Sepsis Guidelines)
```

### 3. Run the Pipeline

```bash
# Run complete pipeline with default settings
python scripts/build_pipeline.py

# Run with custom parameters
python scripts/build_pipeline.py \
    --pubmed-docs 50000 \
    --textbook-docs 10000 \
    --chunk-size 300 \
    --embedding-model "thenlper/gte-small" \
    --max-test-queries 5
```
