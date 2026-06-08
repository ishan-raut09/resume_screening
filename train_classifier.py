import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train():
    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)
    
    # CSV path
    csv_path = os.path.join("dataset", "Resume", "Resume.csv")
    if not os.path.exists(csv_path):
        # Try fallback path if not under Resume subfolder
        csv_path = os.path.join("dataset", "resume_dataset.csv")
        
    print(f"Loading dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please place the dataset CSV there.")
        
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean dataset columns
    # Expected columns: Resume_str, Category
    if "Resume_str" not in df.columns or "Category" not in df.columns:
        # Handle potential capitalization discrepancies
        df.rename(columns={col: col.capitalize() for col in df.columns}, inplace=True)
        if "Resume_str" not in df.columns:
            # Look for alternative name containing resume or text
            text_cols = [c for c in df.columns if "resume" in c.lower() or "text" in c.lower()]
            if text_cols:
                df.rename(columns={text_cols[0]: "Resume_str"}, inplace=True)
        if "Category" not in df.columns:
            cat_cols = [c for c in df.columns if "category" in c.lower() or "label" in c.lower()]
            if cat_cols:
                df.rename(columns={cat_cols[0]: "Category"}, inplace=True)

    # Verify column existence
    if "Resume_str" not in df.columns or "Category" not in df.columns:
        raise ValueError(f"Dataset must contain 'Resume_str' and 'Category' columns. Found columns: {list(df.columns)}")
        
    # Drop rows with nulls in key fields
    df = df.dropna(subset=["Resume_str", "Category"])
    print(f"Dataset shape after cleaning: {df.shape}")
    
    # Print category distribution
    print("\nCategory distribution:")
    print(df["Category"].value_counts())
    
    # Train-test split
    X = df["Resume_str"]
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Define pipeline
    print("\nTraining Linear SVM model...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words='english'
        )),
        ("clf", LinearSVC(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on test data: {acc * 100:.2f}%")
    
    # Save classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    report_path = os.path.join("model", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")
    
    # Save Confusion Matrix
    print("\nGenerating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    categories = sorted(list(set(y)))
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=categories, yticklabels=categories)
    plt.title(f"Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.ylabel("Actual Category")
    plt.xlabel("Predicted Category")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join("model", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # Save pipeline
    pipeline_path = os.path.join("model", "resume_pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model pipeline to {pipeline_path}")
    print("Training completed successfully!")

if __name__ == "__main__":
    train()
