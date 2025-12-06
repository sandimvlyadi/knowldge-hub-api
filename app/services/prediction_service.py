"""
Service untuk ML predictions (Priority, IssueType, Components)
"""

import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer


class PredictionService:
    def __init__(self, model_transformer: SentenceTransformer, 
                 priority_clf, issuetype_clf, component_clf, component_meta):
        self.model_transformer = model_transformer
        self.priority_clf = priority_clf
        self.issuetype_clf = issuetype_clf
        self.component_clf = component_clf
        self.component_mlb = component_meta['mlb']
        self.valid_classes = component_meta['valid_classes']
        self.valid_indices = component_meta['valid_indices']
    
    def predict_priority(self, summary: str, description: str) -> Dict:
        """Prediksi priority dari text"""
        text = f"{summary}. {description}".strip()
        embedding = self.model_transformer.encode([text], convert_to_tensor=True)
        embedding_np = embedding.cpu().numpy()
        
        prediction = self.priority_clf.predict(embedding_np)[0]
        proba = self.priority_clf.predict_proba(embedding_np)[0]
        confidence = float(max(proba))
        
        return {
            "predicted_priority": prediction,
            "confidence": confidence
        }
    
    def predict_issuetype(self, summary: str, description: str) -> Dict:
        """Prediksi issue type dari text"""
        text = f"{summary}. {description}".strip()
        embedding = self.model_transformer.encode([text], convert_to_tensor=True)
        embedding_np = embedding.cpu().numpy()
        
        prediction = self.issuetype_clf.predict(embedding_np)[0]
        proba = self.issuetype_clf.predict_proba(embedding_np)[0]
        confidence = float(max(proba))
        
        return {
            "predicted_issuetype": prediction,
            "confidence": confidence
        }
    
    def predict_components(self, summary: str, description: str) -> Dict:
        """Prediksi components dari text (multi-label)"""
        text = f"{summary}. {description}".strip()
        embedding = self.model_transformer.encode([text], convert_to_tensor=True)
        embedding_np = embedding.cpu().numpy()
        
        prediction_binary = self.component_clf.predict(embedding_np)[0]
        
        # Convert binary ke label names
        predicted_components = [
            self.valid_classes[i] 
            for i, val in enumerate(prediction_binary) 
            if val == 1
        ]
        
        # Get confidence scores (probability)
        try:
            probas = self.component_clf.predict_proba(embedding_np)[0]
            confidence_scores = {
                self.valid_classes[i]: float(probas[i]) 
                for i in range(len(self.valid_classes))
                if prediction_binary[i] == 1
            }
        except:
            # Fallback jika predict_proba tidak tersedia
            confidence_scores = {comp: 1.0 for comp in predicted_components}
        
        return {
            "predicted_components": predicted_components,
            "confidence_scores": confidence_scores
        }
