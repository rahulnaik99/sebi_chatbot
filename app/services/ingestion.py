"""
Ingestion Pipeline
==================
Loads documents → chunks → embeds → indexes into Weaviate Cloud.

Supported file types: PDF, DOCX, TXT
Usage:
    python -m app.services.ingestion --path ./data
    python -m app.services.ingestion --path ./data/loan_policy.pdf
"""

import os
import argparse
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from app.services.get_weaviate import get_weaviate_client
from app.services.get_model import get_embeddings
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_documents(path: str) -> List[Document]:
    """Load all supported documents from a file or directory."""
    p = Path(path)
    files = []

    if p.is_file():
        files = [p]
    elif p.is_dir():
        for ext in ["*.pdf", "*.docx", "*.txt"]:
            files.extend(p.rglob(ext))
    else:
        raise ValueError(f"Path not found: {path}")

    docs = []
    for f in files:
        log.info(f"Loading: {f.name}")
        ext = f.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(f))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(f))
        elif ext == ".txt":
            loader = TextLoader(str(f), encoding="utf-8")
        else:
            log.warning(f"Skipping unsupported file: {f.name}")
            continue
        loaded = loader.load()
        # Tag source metadata
        for doc in loaded:
            doc.metadata["source"] = f.name
        docs.extend(loaded)

    log.info(f"Loaded {len(docs)} raw document pages from {len(files)} files.")
    return docs


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(f"Split into {len(chunks)} chunks.")
    return chunks


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def index_documents(chunks: List[Document]) -> WeaviateVectorStore:
    """Embed and index chunks into Weaviate Cloud."""
    client = get_weaviate_client()
    embeddings = get_embeddings()

    log.info(f"Indexing {len(chunks)} chunks into Weaviate collection '{settings.WEAVIATE_INDEX_NAME}'...")

    vectorstore = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=settings.WEAVIATE_INDEX_NAME,
        text_key="text",
    )

    log.info("Indexing complete.")
    return vectorstore


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_ingestion(path: str) -> WeaviateVectorStore:
    docs = load_documents(path)
    chunks = chunk_documents(docs)
    vectorstore = index_documents(chunks)
    return vectorstore


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoanIQ Ingestion Pipeline")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    args = parser.parse_args()
    run_ingestion(args.path)
    log.info("Done.")
