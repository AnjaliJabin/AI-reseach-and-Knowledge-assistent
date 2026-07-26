import uuid
import os
import logging
from datetime import datetime
from typing import List
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# In-memory document store
documents_db = {}

# ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def extract_text_from_pdf(file_path: str) -> List[dict]:
    pages = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def process_document(doc_id: str, file_path: str, filename: str):
    try:
        documents_db[doc_id]["status"] = "processing"
        pages = extract_text_from_pdf(file_path)
        total_pages = len(pages)

        chunks = []
        chunk_metadatas = []
        chunk_ids = []

        for page_data in pages:
            page_chunks = text_splitter.split_text(page_data["text"])
            for chunk in page_chunks:
                chunk_id = str(uuid.uuid4())
                chunks.append(chunk)
                chunk_metadatas.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_data["page"]
                })
                chunk_ids.append(chunk_id)

        # Batch embed
        embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False).tolist()

        # Store in ChromaDB in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.add(
                documents=chunks[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=chunk_metadatas[i:i+batch_size],
                ids=chunk_ids[i:i+batch_size]
            )

        documents_db[doc_id].update({
            "status": "completed",
            "total_pages": total_pages,
            "total_chunks": len(chunks)
        })
        logger.info(f"Processed {filename}: {len(chunks)} chunks from {total_pages} pages")

    except Exception as e:
        documents_db[doc_id]["status"] = "failed"
        logger.error(f"Failed to process {filename}: {e}")


def semantic_search(query: str, doc_ids: List[str] = None, top_k: int = 5) -> List[dict]:
    query_embedding = embedder.encode([query]).tolist()

    where_filter = None
    if doc_ids:
        if len(doc_ids) == 1:
            where_filter = {"doc_id": doc_ids[0]}
        else:
            where_filter = {"doc_id": {"$in": doc_ids}}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "content": doc,
                "filename": meta.get("filename", ""),
                "page": meta.get("page", 0),
                "doc_id": meta.get("doc_id", ""),
                "score": round(1 - dist, 4)
            })
    return output


def delete_document_chunks(doc_id: str):
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
