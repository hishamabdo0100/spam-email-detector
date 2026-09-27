---
title: Spam Email Detector
emoji: 📧
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Spam Email Detector

TF-IDF + Logistic Regression pipeline (Test Accuracy: 99.47%) with a Gradio
web UI and an auto-generated REST API.

## Files
- `app.py` — Gradio app (UI + API)
- `train.py` — retrains the pipeline from `emails.csv` and saves `model.pkl`
- `model.pkl` — the trained pipeline (vectorizer + model together)
- `emails.csv` — training data (only needed if you re-run `train.py`)
- `requirements.txt` — pinned dependencies

## Deploy on Hugging Face Spaces (recommended, free)

1. Go to https://huggingface.co/new-space
2. Create a Space:
   - **SDK**: Gradio
   - Name it whatever you like (e.g. `spam-email-detector`)
3. Upload these 4 files to the Space (via the web UI "Files" tab, or git):
   - `app.py`
   - `model.pkl`
   - `requirements.txt`
   - `README.md` (optional, but keeps the Space's metadata card correct)
   - You do **not** need to upload `emails.csv` or `train.py` unless you want
     others to be able to retrain it.
4. Wait ~1-2 minutes for the Space to build. It will show a live URL like:
   `https://huggingface.co/spaces/<your-username>/spam-email-detector`

That page IS the web UI. To use it as an API instead of the UI, scroll to the
bottom of the Space page and click **"Use via API"** — Gradio gives you
ready-to-copy Python/JS/cURL snippets that call the same model. In short, once
deployed you can POST to it like:

```python
from gradio_client import Client

client = Client("<your-username>/spam-email-detector")
result = client.predict("Congratulations! You won a free prize!", api_name="/predict")
print(result)
```

## Run locally first (optional, to double check)

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL it prints (usually http://127.0.0.1:7860).

## Retraining the model

If you want to change hyperparameters or retrain on new data:

```bash
pip install -r requirements.txt
python train.py
```

This regenerates `model.pkl` using the same TF-IDF + Logistic Regression
settings found via GridSearchCV in the original notebook (C=100, ngram
range (1,2), min_df=5, max_df=0.7).

## Note on the original notebook

The notebook's final model was labeled "LinearSVC" but due to a small bug
(`model_linearsvc_grid` was accidentally fit using `model_log_pipeline`
instead of `model_linearsvc_pipeline`), the model actually deployed here is
**Logistic Regression** with the winning hyperparameters — this is exactly
the model whose 99.47% test accuracy you saw in the notebook's final cells,
just correctly reproduced and saved as a portable artifact.
