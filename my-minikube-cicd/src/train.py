import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
import pickle

# ---------------------------
# MLflow setup
# ---------------------------
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("sms_spam_detection")


def main():

    # -------------------------------
    # Load Dataset
    # -------------------------------
    df = pd.read_csv('SMSSpamCollection', encoding='latin-1', sep='\t', names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -------------------------------
    # TF-IDF Vectorizer
    # -------------------------------
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # -------------------------------
    # Model Training
    # -------------------------------
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # -------------------------------
    # MLflow Logging
    # -------------------------------
    with mlflow.start_run():

        # Parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("vectorizer", "TFIDF")
        mlflow.log_param("ngram_range", "(1,2)")

        # Metrics
        mlflow.log_metric("accuracy", acc)

        # Save Model and Vectorizer Locally
        joblib.dump(model, 'spam_model.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

        # Log to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
