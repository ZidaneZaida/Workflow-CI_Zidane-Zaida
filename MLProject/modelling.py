import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

def train_ci_model(data_path: str):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"[CI-TRAINING] Model berhasil dilatih! Test Accuracy: {score:.4f}")

    # Simpan juga artefak model ke folder statis agar mudah digunakan untuk Kriteria 4 & Skilled CI
    output_model_dir = "saved_model"
    if os.path.exists(output_model_dir):
        import shutil
        shutil.rmtree(output_model_dir)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.save_model(
        sk_model=model,
        path=output_model_dir,
        signature=signature,
        input_example=X_train.iloc[:5]
    )
    print(f"[CI-TRAINING] Artefak model tersimpan di: MLProject/{output_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
    args = parser.parse_args()
    train_ci_model(args.data_path)
