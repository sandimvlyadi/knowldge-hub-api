"""
Service untuk recommendation menggunakan similarity search
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch


class RecommendationService:
    def __init__(self, model_transformer: SentenceTransformer, df: pd.DataFrame, 
                 issue_embeddings: np.ndarray):
        self.model_transformer = model_transformer
        self.df = df
        self.issue_embeddings = issue_embeddings
    
    def get_recommendations(self, features: List[Dict], top_n: int = 5, 
                          blacklists: List[str] = None) -> List[Dict]:
        """
        Mencari rekomendasi method untuk setiap feature berdasarkan similarity
        """
        if blacklists is None:
            blacklists = ["java.io", "java.util", "java.lang", "org.apache.maven", 
                         "org.junit", "org.slf4j", "org.springframework", "java.net"]
        
        results = []
        
        # Prepare features dataframe
        df_feat = pd.DataFrame(features)
        df_feat['text'] = (df_feat['summary'].fillna('') + ' ' + 
                          df_feat['description'].fillna('')).str.strip()
        
        # Generate embeddings untuk features
        feature_embeddings = self.model_transformer.encode(
            df_feat['text'].tolist(), 
            convert_to_tensor=True
        )
        
        # Pastikan semua tensor di CPU
        if isinstance(feature_embeddings, torch.Tensor):
            feature_embeddings = feature_embeddings.cpu()
        
        # Convert issue_embeddings to CPU tensor
        if isinstance(self.issue_embeddings, torch.Tensor):
            issue_embeddings_cpu = self.issue_embeddings.cpu()
        else:
            issue_embeddings_cpu = torch.tensor(self.issue_embeddings, device='cpu')
        
        # Calculate similarity
        sim_scores_tensor = util.pytorch_cos_sim(
            issue_embeddings_cpu, 
            feature_embeddings
        )
        sim_scores = sim_scores_tensor.cpu().numpy()
        
        # Process untuk setiap feature
        for i in range(len(df_feat)):
            feature_key = df_feat.iloc[i]['key']
            
            # Sort by similarity score
            scores = sim_scores[:, i]
            top_indices = np.argsort(scores)[-top_n:][::-1]
            
            recommendations = []
            all_methods = []
            
            for idx in top_indices:
                issue_key = self.df.iloc[idx]['key']
                score = float(scores[idx])
                methods_raw = self.df.iloc[idx]['methods']
                
                # Filter methods
                if isinstance(methods_raw, list):
                    methods = [
                        m for m in methods_raw 
                        if not any(m.startswith(b) for b in blacklists)
                    ]
                else:
                    methods = []
                
                all_methods.extend(methods)
                
                recommendations.append({
                    "key": issue_key,
                    "score": score,
                    "methods": methods
                })
            
            # Get unique methods
            unique_methods = list(dict.fromkeys(all_methods))
            unique_methods = [
                m for m in unique_methods 
                if not any(m.startswith(b) for b in blacklists)
            ]
            
            results.append({
                "feature_key": feature_key,
                "recommendations": recommendations,
                "unique_methods": unique_methods
            })
        
        return results
