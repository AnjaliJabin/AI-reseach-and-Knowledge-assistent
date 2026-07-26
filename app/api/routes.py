import uuid
import os
import logging
import threading
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.core.config import UPLOAD_DIR
from app.services.document_service import documents_db, process_document, delete_document_chunks, semantic_search
from app.services.ml_service import classify_document, train_classifier
from app.services.ai_service import get_answer, compare_documents, summarize_document

router = APIRouter()
logger = logging.getLogger(__name__)

analytics = {
    "total_questions": 0,
    "query_counts": {}
}


class QuestionRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    session_id: Optional[str] = "default"

class CompareRequest(BaseModel):
    doc_ids: List[str]
    aspect: Optional[str] = "general comparison"

class SummarizeRequest(BaseModel):
    doc_id: str
    summary_type: Optional[str] = "executive"

class SearchRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = 5


@router.post("/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    documents_db[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "file_path": file_path,
        "upload_timestamp": datetime.now().isoformat(),
        "total_pages": 0,
        "total_chunks": 0,
        "status": "pending",
        "category": None
    }

    background_tasks.add_task(_process_and_classify, doc_id, file_path, file.filename)
    return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}


def _process_and_classify(doc_id: str, file_path: str, filename: str):
    process_document(doc_id, file_path, filename)
    if documents_db[doc_id]["status"] == "completed":
        try:
            from app.services.document_service import collection
            results = collection.get(where={"doc_id": doc_id})
            if results["documents"]:
                sample_text = " ".join(results["documents"][:5])
                result = classify_document(sample_text)
                documents_db[doc_id]["category"] = result["category"]
        except Exception as e:
            logger.error(f"Classification failed: {e}")


@router.get("/documents")
def list_documents():
    return {"documents": list(documents_db.values()), "total": len(documents_db)}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    return documents_db[doc_id]


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = documents_db[doc_id].get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    delete_document_chunks(doc_id)
    del documents_db[doc_id]
    return {"message": f"Document {doc_id} deleted successfully"}


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(doc_id: str, background_tasks: BackgroundTasks):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents_db[doc_id]
    delete_document_chunks(doc_id)
    documents_db[doc_id]["status"] = "pending"
    background_tasks.add_task(_process_and_classify, doc_id, doc["file_path"], doc["filename"])
    return {"message": "Reprocessing started", "doc_id": doc_id}


@router.post("/search")
def search(req: SearchRequest):
    results = semantic_search(req.query, req.doc_ids, req.top_k)
    analytics["query_counts"][req.query] = analytics["query_counts"].get(req.query, 0) + 1
    return {"query": req.query, "results": results, "total": len(results)}


@router.post("/ask")
def ask_question(req: QuestionRequest):
    analytics["total_questions"] += 1
    analytics["query_counts"][req.question] = analytics["query_counts"].get(req.question, 0) + 1
    result = get_answer(req.question, req.doc_ids, req.session_id)
    return result


@router.post("/compare")
def compare(req: CompareRequest):
    if len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 documents required")
    return compare_documents(req.doc_ids, req.aspect)


@router.post("/summarize")
def summarize(req: SummarizeRequest):
    if req.doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    return summarize_document(req.doc_id, req.summary_type)


@router.post("/classify/{doc_id}")
def classify(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    from app.services.document_service import collection
    results = collection.get(where={"doc_id": doc_id})
    if not results["documents"]:
        raise HTTPException(status_code=400, detail="Document not yet processed")
    sample_text = " ".join(results["documents"][:5])
    result = classify_document(sample_text)
    documents_db[doc_id]["category"] = result["category"]
    return result


@router.post("/ml/train")
def train_model():
    train_classifier()
    return {"message": "Model trained and saved successfully"}


@router.get("/analytics")
def get_analytics():
    from app.services.document_service import collection
    total_chunks = collection.count()
    completed_docs = [d for d in documents_db.values() if d["status"] == "completed"]
    total_embeddings = sum(d.get("total_chunks", 0) for d in completed_docs)
    top_queries = sorted(analytics["query_counts"].items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_documents": len(documents_db),
        "completed_documents": len(completed_docs),
        "total_chunks_in_db": total_chunks,
        "total_embeddings": total_embeddings,
        "total_questions_answered": analytics["total_questions"],
        "top_queries": [{"query": q, "count": c} for q, c in top_queries]
    }
