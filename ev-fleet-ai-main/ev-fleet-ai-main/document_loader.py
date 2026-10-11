"""ChromaDB ingestion for company documents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
COLLECTION_NAME = "ev_fleet_docs"


def _embeddings(google_api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key,
    )


def get_vector_store(google_api_key: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=_embeddings(google_api_key),
    )


def is_vector_store_empty(google_api_key: str) -> bool:
    try:
        store = get_vector_store(google_api_key)
        return store._collection.count() == 0
    except Exception:
        return True


def ingest_file(file_path: str, google_api_key: str) -> int:
    """Chunk and ingest a .txt or .pdf file. Returns the number of chunks added."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError("Only .txt and .pdf files are supported.")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return 0

    store = get_vector_store(google_api_key)
    store.add_documents(chunks)
    if hasattr(store, "persist"):
        store.persist()
    return len(chunks)


def ingest_uploaded_bytes(
    filename: str,
    data: bytes,
    google_api_key: str,
    tmp_dir: Optional[str] = None,
) -> int:
    """Persist an uploaded Streamlit file to disk, then ingest it."""
    directory = Path(tmp_dir or os.path.join(os.path.dirname(__file__), "uploads"))
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / Path(filename).name
    dest.write_bytes(data)
    return ingest_file(str(dest), google_api_key)
