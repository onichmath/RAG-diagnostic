"""
Build FAISS vector store using LangChain.
"""

import json
from pathlib import Path
from datasets import load_from_disk
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)



def build_faiss_index(
    corpus_dir: Path = Path("data/corpus_raw"),
    corpus_norm_dir: Path = Path("data/corpus_norm"),
    output_dir: Path = Path("data/indices"),
    embedding_model: str = "thenlper/gte-small",
    batch_size: int = 1000,
    use_gpu: bool = True
) -> Path:
    """
    Build FAISS index from all corpus data.
    
    Args:
        corpus_dir: Directory containing raw datasets
        corpus_norm_dir: Directory containing processed guidelines
        output_dir: Directory to save FAISS index
        embedding_model: HuggingFace embedding model
        batch_size: Number of texts to process in each batch
        use_gpu: Whether to use GPU for embeddings
    
    Returns:
        Path to saved FAISS index
    """
    logger.info("Building FAISS index...")
    
    # Load all texts
    all_texts = []
    all_metadata = []
    
    # Load MedRAG datasets from disk
    for dataset_path in corpus_dir.glob("*_*"):
        if dataset_path.is_dir():
            logger.info(f"Loading dataset from {dataset_path.name}...")
            
            dataset = load_from_disk(str(dataset_path))
            
            for item in dataset:
                # Extract text content
                text = item.get("contents") or item.get("text") or item.get("content") or ""
                if text.strip():  # Only add non-empty texts
                    all_texts.append(text)
                    all_metadata.append({
                        "doc_id": item.get("id") or f"{dataset_path.name}_{len(all_texts)}",
                        "source": dataset_path.name.split('_')[0],
                        "title": item.get("title", "")
                    })
    
    # Load processed guidelines from corpus_norm
    guidelines_path = corpus_norm_dir / "guidelines_processed"
    if guidelines_path.exists():
        logger.info("Loading processed guidelines...")
        guidelines_dataset = load_from_disk(str(guidelines_path))
        
        for item in guidelines_dataset:
            text = item.get("contents") or ""
            if text.strip():
                all_texts.append(text)
                all_metadata.append({
                    "doc_id": item.get("id"),
                    "source": item.get("source", "guideline"),
                    "title": item.get("title", ""),
                    "doc_id_orig": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id")
                })
    
    logger.info(f"Loaded {len(all_texts)} texts for indexing")
    
    # Determine device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device}
    )
    logger.info("✅ Embedding model loaded successfully")
    
    # Build FAISS index with batch processing
    logger.info(f"Building FAISS index with batch size {batch_size} (this may take a while)...")
    
    # Process in batches to avoid memory issues and improve speed
    vectorstore = None
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    
    # Use tqdm for progress tracking
    with tqdm(total=len(all_texts), desc="Building FAISS index", unit="texts") as pbar:
        for i in range(0, len(all_texts), batch_size):
            batch_num = (i // batch_size) + 1
            end_idx = min(i + batch_size, len(all_texts))
            
            batch_texts = all_texts[i:end_idx]
            batch_metadata = all_metadata[i:end_idx]
            
            if vectorstore is None:
                # Create initial vectorstore with first batch
                vectorstore = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadata
                )
            else:
                # Add subsequent batches to existing vectorstore
                batch_vectors = embeddings.embed_documents(batch_texts)
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadata,
                    embeddings=batch_vectors
                )
            
            pbar.update(len(batch_texts))
    
    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index"
    vectorstore.save_local(str(index_path))
    
    logger.info(f"FAISS index saved to {index_path}")
    return index_path


def load_faiss_index(
    index_path: Path = Path("data/indices/faiss_index"),
    embedding_model: str = "thenlper/gte-small"
) -> FAISS:
    """Load existing FAISS index."""
    logger.info(f"Loading FAISS index from {index_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device}
    )
    
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    logger.info("FAISS index loaded successfully")
    return vectorstore


def build_faiss_index_fast(
    corpus_dir: Path = Path("data/corpus_raw"),
    corpus_norm_dir: Path = Path("data/corpus_norm"),
    output_dir: Path = Path("data/indices"),
    embedding_model: str = "thenlper/gte-small",
    batch_size: int = 2000,
    use_gpu: bool = True,
    faiss_index_type: str = "IVF"
) -> Path:
    """
    Build FAISS index with optimized performance.
    
    Args:
        corpus_dir: Directory containing raw datasets
        corpus_norm_dir: Directory containing processed guidelines
        output_dir: Directory to save FAISS index
        embedding_model: HuggingFace embedding model
        batch_size: Number of texts to process in each batch
        use_gpu: Whether to use GPU for embeddings
        faiss_index_type: Type of FAISS index ("IVF", "HNSW", "Flat")
    
    Returns:
        Path to saved FAISS index
    """
    import faiss
    
    logger.info("Building optimized FAISS index...")
    
    # Load all texts (same as before)
    all_texts = []
    all_metadata = []
    
    # Load MedRAG datasets from disk
    for dataset_path in corpus_dir.glob("*_*"):
        if dataset_path.is_dir():
            logger.info(f"Loading dataset from {dataset_path.name}...")
            dataset = load_from_disk(str(dataset_path))
            
            for item in dataset:
                text = item.get("contents") or item.get("text") or item.get("content") or ""
                if text.strip():
                    all_texts.append(text)
                    all_metadata.append({
                        "doc_id": item.get("id") or f"{dataset_path.name}_{len(all_texts)}",
                        "source": dataset_path.name.split('_')[0],
                        "title": item.get("title", "")
                    })
    
    # Load processed guidelines
    guidelines_path = corpus_norm_dir / "guidelines_processed"
    if guidelines_path.exists():
        logger.info("Loading processed guidelines...")
        guidelines_dataset = load_from_disk(str(guidelines_path))
        
        for item in guidelines_dataset:
            text = item.get("contents") or ""
            if text.strip():
                all_texts.append(text)
                all_metadata.append({
                    "doc_id": item.get("id"),
                    "source": item.get("source", "guideline"),
                    "title": item.get("title", ""),
                    "doc_id_orig": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id")
                })
    
    logger.info(f"Loaded {len(all_texts)} texts for indexing")
    
    # Initialize embeddings
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device}
    )
    
    # Get embedding dimension from a sample
    sample_embedding = embeddings.embed_query("sample text")
    dimension = len(sample_embedding)
    logger.info(f"Embedding dimension: {dimension}")
    
    # Create FAISS index based on type
    if faiss_index_type == "IVF":
        # IVF (Inverted File) - good balance of speed and memory
        nlist = min(4096, len(all_texts) // 100)  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    elif faiss_index_type == "HNSW":
        # HNSW (Hierarchical Navigable Small World) - very fast search
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the M parameter
    else:  # Flat
        # Flat index - exact search, slower but most accurate
        index = faiss.IndexFlatL2(dimension)
    
    # Process texts in batches and build index
    all_vectors = []
    
    with tqdm(total=len(all_texts), desc="Generating embeddings", unit="texts") as pbar:
        for i in range(0, len(all_texts), batch_size):
            end_idx = min(i + batch_size, len(all_texts))
            batch_texts = all_texts[i:end_idx]
            
            # Generate embeddings for batch
            batch_vectors = embeddings.embed_documents(batch_texts)
            all_vectors.extend(batch_vectors)
            
            pbar.update(len(batch_texts))
    
    # Convert to numpy array
    vectors_array = np.array(all_vectors).astype('float32')
    logger.info(f"Generated embeddings array shape: {vectors_array.shape}")
    
    # Train index if needed (for IVF)
    if faiss_index_type == "IVF":
        logger.info("Training IVF index...")
        index.train(vectors_array)
    
    # Add vectors to index
    logger.info("Adding vectors to FAISS index...")
    index.add(vectors_array)
    
    # Save index and metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index"
    
    # Save FAISS index
    faiss.write_index(index, str(index_path / "index.faiss"))
    
    # Save metadata
    metadata_path = index_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Save index info
    info = {
        "embedding_model": embedding_model,
        "dimension": dimension,
        "index_type": faiss_index_type,
        "num_vectors": len(all_vectors),
        "device": device
    }
    
    info_path = index_path / "index_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Optimized FAISS index saved to {index_path}")
    return index_path
