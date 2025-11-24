"""
Build FAISS vector store using LangChain.
"""

import json
from pathlib import Path
from collections import Counter

from datasets import load_from_disk
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Heuristic filtering / labeling for PubMed chunks
# -------------------------------------------------------

ANIMAL_KEYWORDS = [
    " rat ", " rats ", " mouse ", " mice ", " murine ",
    " rabbit ", " rabbits ", " canine ", " dog ", " dogs ",
    " cat ", " cats ", " monkey ", " monkeys ", " macaque ", " marmoset ",
    " hamster ", " guinea pig ", "guinea-pig",
    " chicken ", " chicks ", " turkey ", " turkeys ",
    " sheep ", " goat ", " pigs ", " pig ", " swine ",
    "sprague-dawley", "wistar", "c57bl/6", "c57bl6"
]

MICROBIO_ORG_KEYWORDS = [
    "bacillus ", "b. thuringiensis", "thuringiensis",
    "streptococcus", "staphylococcus", "lactobacillus", "lactobacilli",
    "escherichia coli", "e. coli", "salmonella", "serratia",
    "pseudomonas", "clostridium", "mycobacterium", "candida",
    "yeast", "fungus", "fungi", "bacteriophage"
]

MICROBIO_LAB_KEYWORDS = [
    "culture medium", "agar", "broth", "fermentor", "fermentors",
    "submerged cultivation", "colony-forming", "cfu",
    "in vitro", "in-vitro", "in vitro experiments"
]

BASIC_SCIENCE_BIOCHEM_KEYWORDS = [
    "ribonucleic acid", "rna polymerase", "dna polymerase",
    "polyadenylate", "poly(a)", "poly(u)", "poly(c)",
    "microsomes", "microsomal", "sarcoplasmic reticulum",
    "mitochondria", "mitochondrial", "enzyme activity",
    "phospholipase", "lipoprotein lipase", "fatty acid synthetase",
    "phosphoenolpyruvate carboxykinase",
    "millipore filter", "radioactivity", "[32p]", "[14c]", "labelled"
]

HUMAN_CLINICAL_HINTS = [
    " patient", " patients", " subjects", " clinical",
    "burned patients", "case report", "a patient is reported",
    "we describe a patient", "severely burned"
]


def contains_any(text: str, keywords) -> bool:
    t = " " + text.lower() + " "
    return any(kw in t for kw in keywords)


def is_animal_study(text: str) -> bool:
    return contains_any(text, ANIMAL_KEYWORDS)


def is_microbiology_basic(text: str) -> bool:
    lower = text.lower()
    if any(h in lower for h in HUMAN_CLINICAL_HINTS):
        return False
    return contains_any(lower, MICROBIO_ORG_KEYWORDS) or contains_any(lower, MICROBIO_LAB_KEYWORDS)


def is_basic_science_biochem(text: str) -> bool:
    lower = text.lower()
    if any(h in lower for h in HUMAN_CLINICAL_HINTS):
        return False
    return contains_any(lower, BASIC_SCIENCE_BIOCHEM_KEYWORDS)


def get_pubmed_label(text: str) -> str:
    if is_animal_study(text):
        return "animal"
    if is_microbiology_basic(text):
        return "microbiology_basic"
    if is_basic_science_biochem(text):
        return "basic_science_biochem"
    return "keep"


DROP_LABELS = {"animal", "microbiology_basic", "basic_science_biochem"}


# -------------------------------------------------------
# FAISS builders
# -------------------------------------------------------

def build_faiss_index(
    corpus_dir: Path = Path("data/corpus_raw"),
    corpus_norm_dir: Path = Path("data/corpus_norm"),
    output_dir: Path = Path("data/indices"),
    embedding_model: str = "thenlper/gte-small",
    batch_size: int = 1000,
    use_gpu: bool = True,
    filter_pubmed: bool = False,
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

    If filter_pubmed=True, applies heuristic filtering on PubMed chunks:
      - drops animal / basic microbiology / basic biochem chunks.
    """
    logger.info("Building FAISS index...")

    # Load all texts
    all_texts = []
    all_metadata = []
    pubmed_filter_counts = Counter()

    for dataset_path in corpus_dir.glob("*_*"):
        if not dataset_path.is_dir():
            continue

        logger.info(f"Loading dataset from {dataset_path.name}...")
        dataset = load_from_disk(str(dataset_path))
        source = dataset_path.name.split('_')[0]  # e.g. "pubmed" or "textbooks"

        for item in dataset:
            text = item.get("contents") or item.get("text") or item.get("content") or ""
            if not text.strip():
                continue

            meta_extra = {}

            # --- PubMed-only filtering (optional) ---
            if filter_pubmed and source == "pubmed":
                label = get_pubmed_label(text)
                if label in DROP_LABELS:
                    pubmed_filter_counts[label] += 1
                    continue  # drop this chunk
                meta_extra["pubmed_label"] = label

            all_texts.append(text)
            meta = {
                "doc_id": item.get("id") or f"{dataset_path.name}_{len(all_texts)}",
                "source": source,
                "title": item.get("title", "")
            }
            meta.update(meta_extra)
            all_metadata.append(meta)

    if filter_pubmed and pubmed_filter_counts:
        logger.info("PubMed filtering stats (dropped chunks):")
        for label, count in pubmed_filter_counts.items():
            logger.info(f"  {label}: {count}")

    # guidelines
    guidelines_path = corpus_norm_dir / "guidelines_processed"
    if guidelines_path.exists():
        logger.info("Loading processed guidelines...")
        guidelines_dataset = load_from_disk(str(guidelines_path))

        for item in guidelines_dataset:
            text = item.get("contents") or ""
            if not text.strip():
                continue

            all_texts.append(text)
            all_metadata.append({
                "doc_id": item.get("id"),
                "source": item.get("source", "guideline"),
                "title": item.get("title", ""),
                "doc_id_orig": item.get("doc_id"),
                "chunk_id": item.get("chunk_id")
            })

    logger.info(f"Loaded {len(all_texts)} texts for indexing (after filtering={filter_pubmed})")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device}
    )
    logger.info("Embedding model loaded successfully")

    logger.info(f"Building FAISS index with batch size {batch_size} (this may take a while)...")

    vectorstore = None

    with tqdm(total=len(all_texts), desc="Building FAISS index", unit="texts") as pbar:
        for i in range(0, len(all_texts), batch_size):
            end_idx = min(i + batch_size, len(all_texts))
            batch_texts = all_texts[i:end_idx]
            batch_metadata = all_metadata[i:end_idx]

            if vectorstore is None:
                vectorstore = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadata
                )
            else:
                batch_vectors = embeddings.embed_documents(batch_texts)
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadata,
                    embeddings=batch_vectors
                )

            pbar.update(len(batch_texts))

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
    faiss_index_type: str = "IVF",
    filter_pubmed: bool = False,
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
        
    Applies the same PubMed filtering if filter_pubmed=True.
    """
    import faiss

    logger.info("Building optimized FAISS index...")

    all_texts = []
    all_metadata = []
    pubmed_filter_counts = Counter()

    for dataset_path in corpus_dir.glob("*_*"):
        if not dataset_path.is_dir():
            continue

        logger.info(f"Loading dataset from {dataset_path.name}...")
        dataset = load_from_disk(str(dataset_path))
        source = dataset_path.name.split('_')[0]

        for item in dataset:
            text = item.get("contents") or item.get("text") or item.get("content") or ""
            if not text.strip():
                continue

            meta_extra = {}

            if filter_pubmed and source == "pubmed":
                label = get_pubmed_label(text)
                if label in DROP_LABELS:
                    pubmed_filter_counts[label] += 1
                    continue
                meta_extra["pubmed_label"] = label

            all_texts.append(text)
            meta = {
                "doc_id": item.get("id") or f"{dataset_path.name}_{len(all_texts)}",
                "source": source,
                "title": item.get("title", "")
            }
            meta.update(meta_extra)
            all_metadata.append(meta)

    if filter_pubmed and pubmed_filter_counts:
        logger.info("PubMed filtering stats (dropped chunks):")
        for label, count in pubmed_filter_counts.items():
            logger.info(f"  {label}: {count}")

    guidelines_path = corpus_norm_dir / "guidelines_processed"
    if guidelines_path.exists():
        logger.info("Loading processed guidelines...")
        guidelines_dataset = load_from_disk(str(guidelines_path))

        for item in guidelines_dataset:
            text = item.get("contents") or ""
            if not text.strip():
                continue

            all_texts.append(text)
            all_metadata.append({
                "doc_id": item.get("id"),
                "source": item.get("source", "guideline"),
                "title": item.get("title", ""),
                "doc_id_orig": item.get("doc_id"),
                "chunk_id": item.get("chunk_id")
            })

    logger.info(f"Loaded {len(all_texts)} texts for indexing (after filtering={filter_pubmed})")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device}
    )

    sample_embedding = embeddings.embed_query("sample text")
    dimension = len(sample_embedding)
    logger.info(f"Embedding dimension: {dimension}")

    if faiss_index_type == "IVF":
        nlist = min(4096, len(all_texts) // 100)  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    elif faiss_index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dimension, 32)
    else:  # Flat
        index = faiss.IndexFlatL2(dimension)

    all_vectors = []

    with tqdm(total=len(all_texts), desc="Generating embeddings", unit="texts") as pbar:
        for i in range(0, len(all_texts), batch_size):
            end_idx = min(i + batch_size, len(all_texts))
            batch_texts = all_texts[i:end_idx]

            batch_vectors = embeddings.embed_documents(batch_texts)
            all_vectors.extend(batch_vectors)

            pbar.update(len(batch_texts))

    vectors_array = np.array(all_vectors).astype('float32')
    logger.info(f"Generated embeddings array shape: {vectors_array.shape}")

    if faiss_index_type == "IVF":
        logger.info("Training IVF index...")
        index.train(vectors_array)

    logger.info("Adding vectors to FAISS index...")
    index.add(vectors_array)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index"

    faiss.write_index(index, str(index_path / "index.faiss"))

    metadata_path = index_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

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
