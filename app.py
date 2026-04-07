import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# SEED (reproducibility)
# =========================
random.seed(42)
np.random.seed(42)

# =========================
# OPENENV YAML SPEC
# =========================
OPENENV_YAML = """
env_name: FakeNewsOpenEnvFinal
type: text_environment
action_space:
  - REAL
  - FAKE
observation_space:
  type: text
difficulty_levels:
  - easy
  - medium
  - hard
reward:
  correct: 1.0
  wrong: -1.0
  confidence_bonus: true
agent_grader:
  metric: accuracy
  scale: 0.0_to_1.0
"""

CONFIG = yaml.safe_load(OPENENV_YAML)

# =========================
# DATASET (3 TASKS REQUIRED BY SPEC)
# =========================
DATA = {
    "easy": [
        ("Government launches education policy", 1),
        ("Scientists discover water on Mars", 1),
        ("Aliens landed in India", 0),
        ("Miracle cure found for all diseases", 0)
    ],
    "medium": [
        ("WHO releases global health report", 1),
        ("Secret lab creates immortality drug", 0),
        ("Economic growth shows mixed trends", 1),
        ("Hidden group controls world politics", 0)
    ],
    "hard": [
        ("AI impacts global employment trends", 1),
        ("Climate report highlights risks", 1),
        ("Unverified cure spreads online rapidly", 0),
        ("Conspiracy group manipulates economy secretly", 0)
    ]
}

# =========================
# MODEL TRAINING (BASELINE)
# =========================
texts, labels = [], []
for lvl in DATA:
    for t, l in DATA[lvl]:
        texts.append(t)
        labels.append(l)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=500)
model.fit(X, labels)

# =========================
# PREDICTION ENGINE
# =========================
def predict(text):
    x = vectorizer.transform([text])
    prob = model.predict_proba(x)[0]

    label = int(np.argmax(prob))
    confidence = float(np.max(prob))

    # partial progress signal (IMPORTANT for OpenEnv)
    confidence += 0.1 * len(set(text.lower().split()) & {"fake", "secret", "conspiracy", "hidden"})
    confidence = min(confidence, 0.99)

    return label, confidence, prob

# =========================
# OPENENV IMPLEMENTATION (FULL SPEC)
# =========================
class OpenEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level = 0
        self.i = 0
        return self.state()

    def state(self):
        lvl = list(DATA.keys())[self.level]
        return DATA[lvl][self.i % len(DATA[lvl])][0]

    def step(self, action, confidence):
        lvl = list(DATA.keys())[self.level]
        true = DATA[lvl][self.i % len(DATA[lvl])][1]

        reward = 1.0 if action == true else -1.0
        reward += (confidence - 0.5) * 0.4

        score = 1.0 if action == true else 0.0

        self.i += 1
        if self.i % 4 == 0 and self.level < 2:
            self.level += 1

        done = self.i >= 12

        return {
            "next_state": None if done else self.state(),
            "reward": reward,
            "score": score,
            "done": done,
            "true": true
        }

# =========================
# INIT ENV
# =========================
env = OpenEnv()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Fake News OpenEnv ", layout="wide")
st.title("🧠 Fake News OpenEnv FINAL ")

mode = st.sidebar.selectbox("Mode", ["Inference", "Evaluation", "Metrics", "Docs"])

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
# EVALUATION (AGENT GRADER REQUIRED)
# =========================
elif mode == "Evaluation":
    state = env.reset()

    y_true, y_pred, rewards = [], [], []

    for _ in range(12):
        pred, conf, _ = predict(state)
        out = env.step(pred, conf)

        y_true.append(out["true"])
        y_pred.append(pred)
        rewards.append(out["reward"])

        if out["done"]:
            break

    st.write("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    st.write("Avg Reward:", round(np.mean(rewards), 3))

    fig, ax = plt.subplots()
    ax.plot(rewards)
    st.pyplot(fig)

# =========================
# METRICS
# =========================
elif mode == "Metrics":
    y_true, y_pred = [], []

    for lvl in DATA:
        for t, l in DATA[lvl]:
            p, _, _ = predict(t)
            y_true.append(l)
            y_pred.append(p)

    st.write(confusion_matrix(y_true, y_pred))

# =========================
# DOCS (FULL SUBMISSION PACKAGE)
# =========================
elif mode == "Docs":
    st.subheader("OpenEnv YAML")
    st.code(OPENENV_YAML, language="yaml")

    st.subheader("requirements.txt")
    st.code("""
streamlit
scikit-learn
matplotlib
numpy
pyyaml
""")

    st.subheader("Dockerfile (HuggingFace / Deployment Ready)")
    st.code("""
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
""")

    st.subheader("README.md")
    st.markdown("""
## Fake News OpenEnv FINAL
### Features
- OpenEnv RL-style environment
- 3 difficulty levels (easy/medium/hard)
- Agent grading (0–1 score)
- Reward shaping with confidence
- ML baseline model
- Streamlit UI
### Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
### HuggingFace Spaces Deployment
- Upload files
- Use Dockerfile OR Streamlit runtime
- App runs on port 7860
""")

# =========================
# BASELINE SCRIPT (REPRODUCIBLE)
# =========================
# run: python inference.py
