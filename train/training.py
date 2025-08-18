import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
from utils.exception import CustomException
from utils.logger import logger


def load_data():
    try:
        logger.info("Loading embeddings...")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
        ARTIFACTS_PATH = os.path.join(BASE_DIR, "..", "artifacts", "embeddings")
        train_embeddings = pd.read_csv(os.path.join(ARTIFACTS_PATH, "train_embeddings.csv"))
        test_embeddings = pd.read_csv(os.path.join(ARTIFACTS_PATH, "test_embeddings.csv"))

        X_train = train_embeddings.drop(columns=['score'])
        y_train = train_embeddings['score']

        X_test = test_embeddings.drop(columns=['score'])
        y_test = test_embeddings['score']

        logger.info("Data loaded successfully.")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error("Error loading data.")
        raise CustomException(e, sys)

def preprocess_data(X_train, X_test):
    try:
        logger.info("Scaling and applying PCA...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        logger.info(f"PCA reduced features from {X_train.shape[1]} to {X_train_pca.shape[1]} components.")

        return X_train_pca, X_test_pca, scaler, pca

    except Exception as e:
        logger.error("Error during preprocessing.")
        raise CustomException(e, sys)

def train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test):
    try:
        logger.info("Training SVM with RBF kernel...")
        svm_rbf = SVC(kernel='rbf', C=0.5, gamma='scale', random_state=42)
        svm_rbf.fit(X_train_pca, y_train)

        y_train_pred = svm_rbf.predict(X_train_pca)
        y_test_pred = svm_rbf.predict(X_test_pca)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        logger.info(f"Training Accuracy: {train_acc:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")

        metrics = {
            "training_accuracy": train_acc,
            "test_accuracy": test_acc,
            "classification_report_test": classification_report(y_test, y_test_pred, output_dict=True)
        }

        return svm_rbf, metrics

    except Exception as e:
        logger.error("Error during model training/evaluation.")
        raise CustomException(e, sys)

def save_artifacts(model, scaler, pca, metrics):
    try:
        logger.info("Saving artifacts...")

        os.makedirs("artifacts/model", exist_ok=True)
        os.makedirs("artifacts/metrics", exist_ok=True)

        # Save model and preprocessing steps
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "pca": pca
        }, "artifacts/model/svm_pipeline.pkl")

        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = f"artifacts/metrics/metrics_{timestamp}.json"
        pd.Series(metrics).to_json(metrics_path)

        logger.info(f"Artifacts saved successfully: {metrics_path}")

    except Exception as e:
        logger.error("Error saving artifacts.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = load_data()
        X_train_pca, X_test_pca, scaler, pca = preprocess_data(X_train, X_test)
        model, metrics = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test)
        save_artifacts(model, scaler, pca, metrics)
        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.error("Pipeline execution failed.")
        raise CustomException(e, sys)