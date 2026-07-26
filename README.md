# AI Research & Knowledge Assistant

A production-grade backend application that enables intelligent document Q&A, semantic search, summarization, comparison, and ML-based classification using RAG (Retrieval-Augmented Generation).

## Architecture

```
Upload PDF → Extract Text → Chunk (RecursiveCharacterTextSplitter) → Embed (all-MiniLM-L6-v2) → Store (ChromaDB)
                                                                                                        ↓
User Query → Embed Query → Semantic Search (ChromaDB) → Retrieve Top-K Chunks → LLM (Gemini) → Answer + Citations
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI 0.140.0 |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB 1.5.9 (persistent) |
| ML Classifier | scikit-learn LogisticRegression + TF-IDF |
| PDF Parsing | PyPDF2 3.0.1 |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

## Setup Instructions

### 1. Clone and install
```bash
git clone <repo-url>
cd AI-Research-Assistant
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Get Gemini API Key
- Go to https://aistudio.google.com/app/apikey
- Create a free API key
- Paste it in `.env` as `GOOGLE_API_KEY=your_key_here`

### 4. Run the server
```bash
python run.py
```

### 5. Open API docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Gemini API key (required) |
| `CHROMA_DB_PATH` | Path to persist ChromaDB (default: ./chroma_db) |
| `UPLOAD_DIR` | Directory for uploaded PDFs (default: ./uploads) |
| `ML_MODEL_PATH` | Path to save/load ML classifier (default: ./ml_model/classifier.pkl) |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload a PDF document |
| GET | `/api/v1/documents` | List all documents |
| GET | `/api/v1/documents/{doc_id}` | Get document details |
| DELETE | `/api/v1/documents/{doc_id}` | Delete a document |
| POST | `/api/v1/documents/{doc_id}/reprocess` | Reprocess a document |
| POST | `/api/v1/search` | Semantic search across documents |
| POST | `/api/v1/ask` | Ask a question (RAG Q&A) |
| POST | `/api/v1/compare` | Compare multiple documents |
| POST | `/api/v1/summarize` | Summarize a document |
| POST | `/api/v1/classify/{doc_id}` | Classify a document |
| POST | `/api/v1/ml/train` | Retrain the ML classifier |
| GET | `/api/v1/analytics` | Get system analytics |

## Chunking Strategy

**RecursiveCharacterTextSplitter** with:
- `chunk_size=1000` characters
- `chunk_overlap=200` characters
- Separators: `["\n\n", "\n", ". ", " ", ""]`

**Why:** Recursive splitting preserves semantic boundaries (paragraphs → sentences → words). The 200-character overlap ensures context is not lost at chunk boundaries, which is critical for accurate retrieval.

## Design Decisions

- **ChromaDB** chosen for persistent local vector storage with cosine similarity
- **all-MiniLM-L6-v2** chosen for fast, high-quality embeddings (384 dimensions)
- **Gemini 1.5 Flash** chosen for free tier availability and strong reasoning
- **scikit-learn** used for classification since TensorFlow doesn't support Python 3.14 yet; LogisticRegression + TF-IDF provides fast, accurate text classification
- **In-memory document store** used for simplicity; can be replaced with SQLite/PostgreSQL

## Limitations

- Only PDF format supported (DOCX/TXT not yet implemented)
- Document store resets on server restart (no persistent DB for metadata)
- TensorFlow replaced with scikit-learn due to Python 3.14 incompatibility
- Large PDFs (>100 pages) may take time to process

## Future Improvements

- Add SQLite/PostgreSQL for persistent document metadata
- Support DOCX and TXT formats
- Add streaming LLM responses
- Implement hybrid search (BM25 + vector)
- Add authentication and multi-user support
- Dockerize the application
