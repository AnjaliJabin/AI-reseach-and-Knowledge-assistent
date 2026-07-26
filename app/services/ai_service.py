import logging
from typing import List, Optional
from google import genai
from app.core.config import GOOGLE_API_KEY
from app.services.document_service import semantic_search

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GOOGLE_API_KEY)


def _generate(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text

# Conversation memory per session
conversation_memory = {}


def get_answer(query: str, doc_ids: List[str] = None, session_id: str = "default") -> dict:
    chunks = semantic_search(query, doc_ids, top_k=5)

    if not chunks:
        return {
            "answer": "I could not find relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "context": []
        }

    context_text = "\n\n".join([
        f"[Source: {c['filename']}, Page {c['page']}]\n{c['content']}"
        for c in chunks
    ])

    history = conversation_memory.get(session_id, [])
    history_text = ""
    if history:
        history_text = "\n".join([f"User: {h['q']}\nAssistant: {h['a']}" for h in history[-3:]])
        history_text = f"\nConversation History:\n{history_text}\n"

    prompt = f"""You are an AI research assistant. Answer the question ONLY based on the provided document context.
If the answer is not in the context, say "I cannot find this information in the provided documents."
Always cite the source document and page number.
{history_text}
Context from documents:
{context_text}

Question: {query}

Provide a clear, accurate answer with citations."""

    response = _generate(prompt)
    answer = response

    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    conversation_memory[session_id].append({"q": query, "a": answer})

    sources = list({(c["filename"], c["page"]) for c in chunks})
    return {
        "answer": answer,
        "sources": [{"filename": s[0], "page": s[1]} for s in sources],
        "context": chunks
    }


def compare_documents(doc_ids: List[str], aspect: str = "general") -> dict:
    all_chunks = []
    for doc_id in doc_ids:
        chunks = semantic_search(aspect, [doc_id], top_k=3)
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"comparison": "No content found for comparison."}

    context = "\n\n".join([
        f"[{c['filename']}, Page {c['page']}]\n{c['content']}"
        for c in all_chunks
    ])

    prompt = f"""Compare the following documents on the topic: "{aspect}".
Provide a structured comparison covering:
- Similarities
- Differences  
- Key points from each document
- Overall conclusion

Documents:
{context}"""

    response = _generate(prompt)
    return {"comparison": response, "sources": all_chunks}


def summarize_document(doc_id: str, summary_type: str = "executive") -> dict:
    chunks = semantic_search("main topics key findings methodology results conclusion", [doc_id], top_k=10)

    if not chunks:
        return {"summary": "No content found to summarize."}

    context = "\n\n".join([f"[Page {c['page']}]\n{c['content']}" for c in chunks])
    filename = chunks[0]["filename"] if chunks else "document"

    type_instructions = {
        "executive": "Write a concise executive summary (3-5 sentences) highlighting the main purpose, key findings, and conclusions.",
        "technical": "Write a detailed technical summary covering methodology, technical approach, implementation details, and results.",
        "bullet": "Provide a bullet-point summary with the most important points organized by topic.",
        "takeaways": "List the top 5-7 key takeaways and actionable insights from this document."
    }

    instruction = type_instructions.get(summary_type, type_instructions["executive"])

    prompt = f"""Based on the following content from "{filename}", {instruction}

Content:
{context}"""

    response = _generate(prompt)
    return {
        "summary": response,
        "summary_type": summary_type,
        "filename": filename,
        "sources": chunks
    }
