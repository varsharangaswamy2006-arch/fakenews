import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = [
    ("Government launches education policy", 1),
    ("Scientists discover water on Mars", 1),
    ("Aliens landed in India", 0),
    ("Miracle cure found for all diseases", 0)
]

texts = [t for t, l in DATA]
labels = [l for t, l in DATA]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=500)
model.fit(X, labels)

def predict(text):
    x = vectorizer.transform([text])
    prob = model.predict_proba(x)[0]
    label = int(np.argmax(prob))
    confidence = float(np.max(prob))

    return {
        "label": "REAL" if label == 1 else "FAKE",
        "confidence": round(confidence, 3)
    }

if __name__ == "__main__":
    text = sys.argv[1]
    print(json.dumps(predict(text)))
