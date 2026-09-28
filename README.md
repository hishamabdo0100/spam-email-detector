# 📧 Spam Email Detector

TF-IDF + Logistic Regression pipeline (test accuracy: 99.47%) served with
FastAPI: a web UI at `/` and a JSON API at `/predict`.

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (API docs).

## API
```bash
curl -X POST https://YOUR-APP.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free prize!"}'
```
Response:
```json
{"label": "spam", "is_spam": true, "spam_probability": 0.9786, "ham_probability": 0.0214}
```

## Files
- `main.py` - FastAPI app (UI + API)
- `model.pkl` - trained pipeline (vectorizer + classifier)
- `train.py` - retrains the model from `emails.csv`
- `requirements.txt` - pinned dependencies
