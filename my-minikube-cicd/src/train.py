import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn

# MLflow URI from env variable
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("sms_spam_detection")

# Dataset path from env variable
DATA_PATH = os.environ.get("DATA_PATH", "src/SMSSpamCollection")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH, encoding='latin-1', sep='\t', names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Logistic Regression Model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # MLflow Logging
    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("vectorizer", "TFIDF")
        mlflow.log_param("ngram_range", "(1,2)")

        mlflow.log_metric("accuracy", acc)

        # Save locally
        joblib.dump(model, 'spam_model.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

        # Log to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
