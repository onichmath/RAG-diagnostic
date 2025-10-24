# RAG Diagnostic Pipeline

A healthcare domain-specific RAG (Retrieval-Augmented Generation) system for evaluating and improving retrieval performance on medical queries using clinical guidelines, PubMed articles, and medical textbooks.

## Project Overview

This project implements a comprehensive RAG system specifically designed for medical information retrieval, featuring:

- **Multi-source Data**: Clinical guidelines, PubMed articles, medical textbooks
- **Advanced Retrieval**: FAISS vector stores with multiple index types
- **Evaluation**: Precision@k, NCDG@k, Throughput metrics

## Project Structure

```
RAG-diagnostic/
├── data/                    # All data artifacts
│   ├── corpus_raw/         # Raw datasets (MedRAG, guidelines)
│   ├── corpus_norm/         # Processed datasets (chunked)
│   ├── guidelines/         # Clinical guideline PDFs (not in repo)
│   └── requests/           # Test queries and evaluation data
├── src/                    # Source code
│   ├── ingest/            # Data downloading and preprocessing
|   ├── eval/              # Evaluation files
│   └── retriever/         # Vector store and retrieval
├── scripts/               # Executable scripts
│   └── build_pipeline.py  # Main pipeline script
├── notebooks/             # Jupyter notebooks for analysis
├── config/                # Configuration files
├── infra/                 # Infrastructure as code
└── env.yml                # Project Dependencies 
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone git@github.com:onichmath/RAG-diagnostic.git
cd RAG-diagnostic

# Install dependencies
conda env create -f env.yml
```

### 2. Download Clinical Guidelines

**Important**: Clinical guideline PDFs are not included in the repository due to copyright restrictions. You need to download them separately:

```bash
# Download clinical guidelines to data/guidelines/
# Required files:
# - "Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021"
- https://link.springer.com/article/10.1007/s00134-021-06506-y
# - "Introduction and Methodology: Standards of Care in Diabetes—2024"
- https://diabetesjournals.org/care/article/47/Supplement_1/S1/153952/Introduction-and-Methodology-Standards-of-Care-in
# - "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure: A Report of the American College of Cardiology/American Heart Association Joint Committee on Clinical Practice Guidelines"
- https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063
# - "2019 AHA/ACC/HRS Focused Update of the 2014 AHA/ACC/HRS Guideline for the Management of Patients With Atrial Fibrillation"
- https://www.ahajournals.org/doi/10.1161/CIR.0000000000000665
# - "2021 Guideline for the Prevention of Stroke in Patients With Stroke and Transient Ischemic Attack"
- https://www.ahajournals.org/doi/10.1161/STR.0000000000000375
# - "2025 Clinical Practice Guideline Update by the Infectious Diseases Society of America on the Treatment and Management of COVID-19: Infliximab"
- https://pubmed.ncbi.nlm.nih.gov/40831386/
```

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
