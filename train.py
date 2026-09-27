"""
Train the final spam-detection pipeline (TF-IDF + Logistic Regression)
using the exact best hyperparameters found via GridSearchCV in the
original notebook, and save it as a single deployable artifact.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "emails.csv"
MODEL_OUT = "model.pkl"

# 1. Load data
df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True, ignore_index=True)

X = df["text"]
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Build the final pipeline with the best params from GridSearchCV
#    (model__C=100, vectorizer__max_df=0.7, vectorizer__min_df=5, ngram_range=(1,2))
pipeline = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.7,
                sublinear_tf=True,
            ),
        ),
        (
            "model",
            LogisticRegression(
                C=100, max_iter=1000, random_state=42, class_weight="balanced"
            ),
        ),
    ]
)

# 3. Train
pipeline.fit(X_train, y_train)

# 4. Evaluate on held-out test set
y_pred = pipeline.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:     ", precision_score(y_test, y_pred))
print("Recall:        ", recall_score(y_test, y_pred))
print("F1 Score:      ", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Save the whole pipeline (vectorizer + model together) as one artifact
joblib.dump(pipeline, MODEL_OUT)
print(f"\nSaved trained pipeline to {MODEL_OUT}")
