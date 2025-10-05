# src/backend/predict.py
import os
import joblib
import numpy as np

# ------------------ Paths ------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

def load_model():
    """Load trained model and vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError("❌ Model or vectorizer not found. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer

def predict_news(text: str):
    """Predict if news text is REAL or FAKE with probability and confidence."""
    model, vectorizer = load_model()
    x = vectorizer.transform([text])

    # ------------------ Probability ------------------
    try:
        prob = model.predict_proba(x)[0]
        classes = [str(c).upper() for c in model.classes_]

        if "FAKE" in classes:
            fake_index = classes.index("FAKE")
        elif "1" in classes:  # if numeric labels
            fake_index = classes.index("1")
        else:
            fake_index = 0

        prob_fake = float(prob[fake_index])

    except Exception:
        # fallback (for SVM etc.)
        score = model.decision_function(x)[0]
        prob_fake = 1 / (1 + np.exp(-score))

    # ------------------ Decision ------------------
    threshold = 0.5
    label = "FAKE" if prob_fake >= threshold else "REAL"

    # Confidence (0–1 scale, higher = more certain)
    confidence = abs(prob_fake - threshold) * 2
    confidence = min(max(confidence, 0.0), 1.0)

    return {
        "label": label,
        "probability_fake": round(prob_fake, 3),
        "confidence": round(confidence, 3),
    }

# ------------------ Test Run ------------------
if __name__ == "__main__":
    examples = [
        "ISRO launches new satellite for weather monitoring",   # real
        "Taj Mahal to be demolished for shopping mall!",        # fake
    ]
    for text in examples:
        result = predict_news(text)
        print(f"\n📰 Text: {text}")
        print(f"➡️ Predicted: {result['label']} | Probability Fake: {result['probability_fake']} | Confidence: {result['confidence']}")
