"""
Simplified ingest module using LangChain for document loading and processing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd

logger = logging.getLogger(__name__)


class LangChainIngest:
    """Simplified ingest using LangChain document loaders and splitters."""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 60,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize LangChain ingest.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators
        )
    
    def load_pdfs_from_directory(self, directory: Path) -> List[Document]:
        """
        Load all PDFs from a directory using LangChain.
        
        Args:
            directory: Directory containing PDF files
        
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading PDFs from {directory}")
        
        # Use LangChain's DirectoryLoader for PDFs
        loader = DirectoryLoader(
            str(directory),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} PDF documents")
        
        return documents
    
    def load_text_files_from_directory(self, directory: Path) -> List[Document]:
        """
        Load all text files from a directory using LangChain.
        
        Args:
            directory: Directory containing text files
        
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading text files from {directory}")
        
        loader = DirectoryLoader(
            str(directory),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} text documents")
        
        return documents
    
    def chunk_documents(self, documents: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Chunk documents using LangChain's text splitter.
        
        Args:
            documents: List of LangChain Document objects
            metadata: Additional metadata to add to chunks
        
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Add additional metadata if provided
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def documents_to_medrag_format(self, documents: List[Document], source: str = "document") -> List[Dict[str, Any]]:
        """
        Convert LangChain documents to MedRAG format.
        
        Args:
            documents: List of LangChain Document objects
            source: Source identifier for documents
        
        Returns:
            List of dictionaries in MedRAG format
        """
        medrag_docs = []
        
        for i, doc in enumerate(documents):
            # Extract document ID from metadata or create one
            doc_id = doc.metadata.get("source", f"doc_{i:04d}")
            if isinstance(doc_id, Path):
                doc_id = doc_id.stem
            
            medrag_doc = {
                "id": f"{doc_id}_chunk_{i:04d}",
                "contents": doc.page_content,
                "source": source,
                "title": doc.metadata.get("title", doc_id),
                "doc_id": doc_id,
                "chunk_id": i,
                "chunk_size": len(doc.page_content),
                **doc.metadata  # Include all original metadata
            }
            
            medrag_docs.append(medrag_doc)
        
        return medrag_docs
    
    def process_guidelines(
        self,
        guidelines_dir: Path,
        output_dir: Path,
        dataset_name: str = "guidelines_processed"
    ) -> Path:
        """
        Process guidelines using LangChain and save in MedRAG format.
        
        Args:
            guidelines_dir: Directory containing guideline PDFs
            output_dir: Directory to save processed dataset
            dataset_name: Name for the output dataset
        
        Returns:
            Path to saved dataset
        """
        logger.info(f"Processing guidelines from {guidelines_dir}")
        
        documents = self.load_pdfs_from_directory(guidelines_dir)
        #print(documents)
        
        if not documents:
            logger.warning("No PDF documents found")
            return output_dir / dataset_name
        
        chunks = self.chunk_documents(documents, metadata={"source": "guideline"})
        
        medrag_docs = self.documents_to_medrag_format(chunks, source="guideline")
        
        df = pd.DataFrame(medrag_docs)
        dataset = Dataset.from_pandas(df)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / dataset_name
        dataset.save_to_disk(str(output_path))
        
        logger.info(f"Saved {len(medrag_docs)} guideline chunks to {output_path}")
        return output_path
    
    def load_huggingface_dataset(
        self,
        dataset_name: str,
        output_dir: Path,
        max_documents: int = 100000,
        force_download: bool = False
    ) -> Path:
        """
        Load HuggingFace dataset with document limits.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            output_dir: Directory to save the dataset
            max_documents: Maximum number of documents to load
            force_download: Force download even if local copy exists
        
        Returns:
            Path to saved dataset
        """
        dataset_name_clean = dataset_name.split('/')[-1]
        output_path = output_dir / f"{dataset_name_clean}_{max_documents}docs"
        
        if not force_download and output_path.exists():
            try:
                dataset = load_from_disk(str(output_path))
                logger.info(f"✅ Found existing dataset at {output_path} ({len(dataset)} documents)")
                return output_path
            except Exception as e:
                logger.warning(f"⚠️ Found directory {output_path} but it's not a valid dataset: {e}")
        
        logger.info(f"📥 Loading {max_documents:,} documents from {dataset_name} dataset...")
        
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        logger.info("Converting streaming dataset to regular dataset...")
        data_list = []
        for i, item in enumerate(dataset):
            data_list.append(item)
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i:,} documents...")
            
            if i + 1 >= max_documents:
                logger.info(f"Reached document limit of {max_documents:,}")
                break
        
        regular_dataset = Dataset.from_list(data_list)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        regular_dataset.save_to_disk(str(output_path))
        
        logger.info(f"✅ Saved {len(regular_dataset):,} documents to {output_path}")
        return output_path
    
    def load_all_medrag_data(
        self,
        corpus_dir: Path = Path("data/corpus_raw"),
        pubmed_docs: int = 100000,
        textbook_docs: int = 10000,
        force_download: bool = False
    ) -> Dict[str, Path]:
        """
        Load all MedRAG datasets using LangChain approach.
        
        Args:
            corpus_dir: Directory to save datasets
            pubmed_docs: Number of PubMed documents to load
            textbook_docs: Number of textbook documents to load
            force_download: Force download even if local copies exist
        
        Returns:
            Dict with paths to loaded datasets
        """
        corpus_dir.mkdir(parents=True, exist_ok=True)
        loaded = {}
        
        logger.info("🔄 Loading MedRAG datasets...")
        
        pubmed_path = self.load_huggingface_dataset(
            "MedRAG/pubmed",
            corpus_dir,
            max_documents=pubmed_docs,
            force_download=force_download
        )
        loaded["pubmed"] = pubmed_path
        
        textbook_path = self.load_huggingface_dataset(
            "MedRAG/textbooks",
            corpus_dir,
            max_documents=textbook_docs,
            force_download=force_download
        )
        loaded["textbooks"] = textbook_path
        
        logger.info(f"✅ Loaded {len(loaded)} datasets from {corpus_dir}")
        return loaded


def process_guidelines(
    guidelines_dir: Path = Path("data/guidelines"),
    output_dir: Path = Path("data/corpus_norm"),
    chunk_size: int = 300,
    chunk_overlap: int = 60
) -> Path:
    """
    Simple function to process guidelines using LangChain.
    
    Args:
        guidelines_dir: Directory containing guideline PDFs
        output_dir: Directory to save processed dataset
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        Path to saved dataset
    """
    ingest = LangChainIngest(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return ingest.process_guidelines(guidelines_dir, output_dir)


def load_medrag_data(
    corpus_dir: Path = Path("data/corpus_raw"),
    pubmed_docs: int = 100000,
    textbook_docs: int = 10000,
    force_download: bool = False
) -> Dict[str, Path]:
    """
    Simple function to load MedRAG datasets.
    
    Args:
        corpus_dir: Directory to save datasets
        pubmed_docs: Number of PubMed documents to load
        textbook_docs: Number of textbook documents to load
        force_download: Force download even if local copies exist
    
    Returns:
        Dict with paths to loaded datasets
    """
    ingest = LangChainIngest()
    return ingest.load_all_medrag_data(
        corpus_dir=corpus_dir,
        pubmed_docs=pubmed_docs,
        textbook_docs=textbook_docs,
        force_download=force_download
    )


def list_available_datasets(corpus_dir: Path = Path("data/corpus_raw")) -> Dict[str, int]:
    """
    Simple function to list available datasets.
    
    Args:
        corpus_dir: Directory to scan for datasets
    
    Returns:
        Dict with dataset names and document counts
    """
    available = {}
    
    if not corpus_dir.exists():
        return available
    
    for item in corpus_dir.iterdir():
        if item.is_dir() and item.name.endswith("_shards"):
            try:
                dataset = load_from_disk(str(item))
                available[item.name] = len(dataset)
            except Exception as e:
                logger.warning(f"Could not load dataset {item.name}: {e}")
    
    return available
