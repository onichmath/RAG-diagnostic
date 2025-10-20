# Retriever Module

This module handles vector store creation, embedding generation, and document retrieval for the RAG diagnostic pipeline.

## 📁 Module Structure

### `faiss_builder.py`
Builds and manages FAISS vector stores using LangChain and HuggingFace embeddings.

**Functions:**
- `build_faiss_index()`: Standard FAISS index building using LangChain
- `build_faiss_index_fast()`: Optimized FAISS index building with custom index types
- `load_faiss_index()`: Load existing FAISS index

**Key Features:**
- Support for multiple FAISS index types (IVF, HNSW, Flat)
- Batch processing for large datasets
- GPU acceleration support
- Metadata preservation
- Progress tracking with tqdm