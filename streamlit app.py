import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from model import predict

# =========================
# DATA
# =========================
DATA = [
    ("Government launches education policy", 1),
    ("Scientists discover water on Mars", 1),
    ("Aliens landed in India", 0),
    ("Miracle cure found for all diseases", 0)
]

# =========================
# UI
# =========================
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("🧠 Fake News Detector")

mode = st.sidebar.selectbox("Mode", ["Inference", "Evaluation"])

# =========================
# INFERENCE
# =========================
if mode == "Inference":
    text = st.text_area("Enter News")

    if st.button("Predict"):
        pred, conf, prob = predict(text)

        st.success("REAL" if pred else "FAKE")
        st.write("Confidence:", round(conf, 3))

        fig, ax = plt.subplots()
        ax.bar(["FAKE", "REAL"], prob)
        st.pyplot(fig)

# =========================
# EVALUATION
# =========================
elif mode == "Evaluation":
    y_true, y_pred = [], []

    for t, l in DATA:
        p, _, _ = predict(t)
        y_true.append(l)
        y_pred.append(p)

    st.write("Accuracy:", accuracy_score(y_true, y_pred))
    st.write(confusion_matrix(y_true, y_pred))

