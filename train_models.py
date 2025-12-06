"""
Training script untuk semua ML models yang dibutuhkan oleh API.
Script ini akan:
1. Load dan preprocess data dari graphs.json
2. Generate embeddings menggunakan SentenceTransformer
3. Train 3 classifier models: Priority, IssueType, Component
4. Train clustering model (KMeans)
5. Simpan semua models dan artifacts dengan pickle
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, f1_score


def load_json_data(filename):
    """Load JSON data dengan berbagai format"""
    p = Path(filename)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    
    try:
        return pd.read_json(p)
    except ValueError:
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.json_normalize(data)


def create_text_column(df):
    """Bersihkan dan gabungkan summary + description"""
    df['summary'] = df['summary'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    
    df['summary'] = df['summary'].apply(lambda x: x + '.' if x and not x.strip().endswith('.') else x)
    df['description'] = df['description'].apply(lambda x: x + '.' if x and not x.strip().endswith('.') else x)
    
    df['text'] = (df['summary'] + ' ' + df['description']).str.strip()
    return df


def train_models():
    """Main training function"""
    print("="*80)
    print("KNOWLEDGE HUB - MODEL TRAINING")
    print("="*80)
    
    # 1. Load SentenceTransformer Model
    print("\n[1/8] Loading SentenceTransformer model...")
    model_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Load and Prepare Data
    print("[2/8] Loading graphs.json...")
    df = load_json_data("graphs.json")
    df = create_text_column(df)
    print(f"   Loaded {len(df)} records")
    
    # 3. Generate Embeddings
    print("[3/8] Generating embeddings for all issues...")
    issue_embeddings = model_transformer.encode(df['text'].tolist(), convert_to_tensor=True)
    issue_embeddings_np = issue_embeddings.cpu().numpy()
    print(f"   Shape: {issue_embeddings_np.shape}")
    
    # 4. Train Priority Classifier
    print("[4/8] Training Priority Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        issue_embeddings_np, df['priority'], test_size=0.2, random_state=42
    )
    priority_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    priority_clf.fit(X_train, y_train)
    y_pred = priority_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.2f}")
    
    # 5. Train IssueType Classifier
    print("[5/8] Training IssueType Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        issue_embeddings_np, df['issuetype'], test_size=0.2, random_state=42
    )
    issuetype_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    issuetype_clf.fit(X_train, y_train)
    y_pred = issuetype_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.2f}")
    
    # 6. Train Component Predictor (Multi-label)
    print("[6/8] Training Component Predictor...")
    components_data = df['components'].apply(lambda x: x if isinstance(x, list) else [])
    mlb = MultiLabelBinarizer()
    y_raw = mlb.fit_transform(components_data)
    
    # Filter komponen langka
    min_count = 5
    comp_counts = y_raw.sum(axis=0)
    valid_indices = np.where(comp_counts >= min_count)[0]
    y_filtered = y_raw[:, valid_indices]
    valid_classes = mlb.classes_[valid_indices]
    
    print(f"   Components: {len(mlb.classes_)} -> {len(valid_classes)} (filtered, min={min_count})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        issue_embeddings_np, y_filtered, test_size=0.2, random_state=42
    )
    component_clf = OneVsRestClassifier(
        LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000)
    )
    component_clf.fit(X_train, y_train)
    y_pred = component_clf.predict(X_test)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    print(f"   F1-Score (micro): {f1_micro:.2f}")
    
    # 7. Train Clustering Model
    print("[7/8] Training Clustering Model (KMeans)...")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(issue_embeddings_np)
    df['cluster'] = cluster_labels
    print(f"   Created {n_clusters} clusters")
    
    # 8. Save All Models and Artifacts
    print("[8/8] Saving models and artifacts...")
    output_dir = Path("trained_models")
    output_dir.mkdir(exist_ok=True)
    
    # Save models
    with open(output_dir / "priority_classifier.pkl", "wb") as f:
        pickle.dump(priority_clf, f)
    
    with open(output_dir / "issuetype_classifier.pkl", "wb") as f:
        pickle.dump(issuetype_clf, f)
    
    with open(output_dir / "component_classifier.pkl", "wb") as f:
        pickle.dump(component_clf, f)
    
    with open(output_dir / "component_mlb.pkl", "wb") as f:
        pickle.dump({"mlb": mlb, "valid_classes": valid_classes, "valid_indices": valid_indices}, f)
    
    with open(output_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    # Save embeddings dan dataframe
    np.save(output_dir / "issue_embeddings.npy", issue_embeddings_np)
    df.to_pickle(output_dir / "graphs_df.pkl")
    
    # Save SentenceTransformer model name (untuk consistency)
    with open(output_dir / "model_info.json", "w") as f:
        json.dump({
            "transformer_model": "all-MiniLM-L6-v2",
            "n_clusters": n_clusters,
            "training_date": pd.Timestamp.now().isoformat(),
            "total_records": len(df)
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print("\nSaved files:")
    print("  - trained_models/priority_classifier.pkl")
    print("  - trained_models/issuetype_classifier.pkl")
    print("  - trained_models/component_classifier.pkl")
    print("  - trained_models/component_mlb.pkl")
    print("  - trained_models/kmeans_model.pkl")
    print("  - trained_models/issue_embeddings.npy")
    print("  - trained_models/graphs_df.pkl")
    print("  - trained_models/model_info.json")
    print("\nYou can now run the FastAPI server!")


if __name__ == "__main__":
    train_models()
