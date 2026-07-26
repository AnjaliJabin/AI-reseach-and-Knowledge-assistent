from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.services.ml_service import load_classifier
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="AI Research & Knowledge Assistant",
    description="Production-grade RAG-based document intelligence system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    load_classifier()
    logging.getLogger(__name__).info("AI Research Assistant started successfully")


@app.get("/")
def root():
    return {
        "message": "AI Research & Knowledge Assistant API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
