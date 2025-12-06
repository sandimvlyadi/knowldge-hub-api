"""
Service untuk clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sentence_transformers import SentenceTransformer


class ClusteringService:
    def __init__(self, model_transformer: SentenceTransformer, 
                 kmeans_model, df: pd.DataFrame, issue_embeddings: np.ndarray):
        self.model_transformer = model_transformer
        self.kmeans = kmeans_model
        self.df = df
        self.issue_embeddings = issue_embeddings
    
    def predict_cluster(self, summary: str, description: str, n_similar: int = 5) -> Dict:
        """
        Prediksi cluster untuk text baru dan kembalikan similar issues
        """
        # Generate embedding
        text = f"{summary}. {description}".strip()
        embedding = self.model_transformer.encode([text], convert_to_tensor=True)
        embedding_np = embedding.cpu().numpy()
        
        # Predict cluster
        cluster_id = int(self.kmeans.predict(embedding_np)[0])
        
        # Get similar issues dari cluster yang sama
        cluster_issues = self.df[self.df['cluster'] == cluster_id]
        
        # Ambil sample issues dari cluster ini
        if len(cluster_issues) > n_similar:
            similar_issues = cluster_issues.sample(n=n_similar, random_state=42)
        else:
            similar_issues = cluster_issues
        
        similar_list = []
        for _, row in similar_issues.iterrows():
            similar_list.append({
                "key": row['key'],
                "summary": row['summary'],
                "priority": row.get('priority', ''),
                "issuetype": row.get('issuetype', '')
            })
        
        return {
            "cluster_id": cluster_id,
            "similar_issues": similar_list
        }
