"""
Spam Email Detector - Gradio app.

Running this file launches:
  1. A web UI where a user can paste email text and get a prediction.
  2. An automatic REST API (Gradio exposes every function as an API
     endpoint at /run/predict, and gives you ready-to-copy client code
     from the "Use via API" link at the bottom of the app page).
"""

import joblib
import gradio as gr

MODEL_PATH = "model.pkl"
pipeline = joblib.load(MODEL_PATH)

LABELS = {0: "Not Spam (Ham) ✅", 1: "Spam 🚫"}


def predict_email(email_text: str):
    if not email_text or not email_text.strip():
        return "Please enter some email text.", {}

    label = int(pipeline.predict([email_text])[0])

    # Get a confidence-like score from the decision function when available
    confidences = {}
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        proba = pipeline.predict_proba([email_text])[0]
        confidences = {LABELS[0]: float(proba[0]), LABELS[1]: float(proba[1])}
    else:
        confidences = {LABELS[label]: 1.0}

    return LABELS[label], confidences


demo = gr.Interface(
    fn=predict_email,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste the email text here...",
        label="Email content",
    ),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Label(label="Confidence", num_top_classes=2),
    ],
    title="📧 Spam Email Detector",
    description=(
        "Paste an email's text below and the model will classify it as "
        "Spam or Not Spam (Ham). Model: TF-IDF + Logistic Regression."
    ),
    examples=[
        [
            "Congratulations! You have been selected to receive a $1,000 cash "
            "prize. Claim your reward now by clicking the link below. This "
            "exclusive offer expires today, so act fast!"
        ],
        [
            "Hi Ahmed, I wanted to remind you that our project meeting is "
            "scheduled for tomorrow at 10 AM. Please bring the updated report."
        ],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    # server_name="0.0.0.0" makes it reachable when deployed (e.g. Docker,
    # Hugging Face Spaces); share=False is fine on Spaces since HF hosts it.
    demo.launch(server_name="0.0.0.0")
