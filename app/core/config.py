import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "./ml_model/classifier.pkl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

DOCUMENT_CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing",
    "General"
]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs("./ml_model", exist_ok=True)
