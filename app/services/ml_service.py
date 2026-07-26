import os
import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from app.core.config import ML_MODEL_PATH, DOCUMENT_CATEGORIES

logger = logging.getLogger(__name__)

# Training data - keyword-based samples per category
TRAINING_DATA = {
    "Artificial Intelligence": [
        "artificial intelligence AI systems intelligent agents reasoning planning",
        "AI applications machine perception knowledge representation",
        "intelligent systems expert systems AI algorithms decision making",
        "artificial general intelligence narrow AI deep learning neural",
        "AI ethics bias fairness transparency explainability",
    ],
    "Machine Learning": [
        "machine learning supervised unsupervised reinforcement learning algorithms",
        "training data model evaluation overfitting underfitting regularization",
        "gradient descent optimization loss function backpropagation",
        "random forest decision tree ensemble methods boosting bagging",
        "feature engineering dimensionality reduction PCA clustering",
        "cross validation hyperparameter tuning model selection",
    ],
    "Computer Vision": [
        "image recognition object detection convolutional neural network CNN",
        "image segmentation feature extraction visual recognition",
        "YOLO ResNet VGG image classification deep learning vision",
        "optical flow video analysis tracking detection bounding box",
        "image preprocessing augmentation pixel feature maps",
    ],
    "Natural Language Processing": [
        "natural language processing NLP text classification sentiment analysis",
        "tokenization word embeddings BERT GPT transformer language model",
        "named entity recognition POS tagging parsing syntax",
        "machine translation question answering text summarization",
        "LSTM RNN sequence to sequence attention mechanism",
    ],
    "Robotics": [
        "robotics autonomous robot motion planning control systems",
        "robot perception sensors actuators kinematics dynamics",
        "SLAM simultaneous localization mapping path planning",
        "robot arm manipulation grasping trajectory planning",
        "autonomous vehicles drones UAV navigation obstacle avoidance",
    ],
    "Cyber Security": [
        "cybersecurity network security intrusion detection firewall",
        "encryption cryptography authentication authorization access control",
        "malware ransomware phishing vulnerability penetration testing",
        "zero trust security threat intelligence incident response",
        "SIEM SOC security operations center threat hunting",
    ],
    "Cloud Computing": [
        "cloud computing AWS Azure GCP infrastructure as a service",
        "microservices containerization Docker Kubernetes orchestration",
        "serverless functions scalability elasticity load balancing",
        "DevOps CI CD pipeline deployment automation monitoring",
        "distributed systems fault tolerance high availability SLA",
    ],
    "General": [
        "research study analysis methodology results conclusion findings",
        "introduction background literature review related work",
        "data collection experiment evaluation performance metrics",
        "discussion future work limitations acknowledgments references",
        "abstract overview summary paper report document",
    ]
}


def train_classifier():
    texts, labels = [], []
    for category, samples in TRAINING_DATA.items():
        for sample in samples:
            texts.append(sample)
            labels.append(category)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    logger.info(f"Classifier trained:\n{classification_report(y_test, y_pred, zero_division=0)}")

    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
    with open(ML_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Model saved to {ML_MODEL_PATH}")
    return pipeline


def load_classifier():
    if os.path.exists(ML_MODEL_PATH):
        with open(ML_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_classifier()


def classify_document(text: str) -> dict:
    classifier = load_classifier()
    prediction = classifier.predict([text])[0]
    probabilities = classifier.predict_proba([text])[0]
    classes = classifier.classes_

    scores = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "category": prediction,
        "confidence": round(float(max(probabilities)), 4),
        "top_categories": [{"category": k, "score": v} for k, v in top_scores]
    }
