"""
Main FastAPI application
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from app.models.schemas import (
    RecommendationRequest, RecommendationResponse,
    TextPredictionRequest, PriorityPredictionResponse,
    IssueTypePredictionResponse, ComponentPredictionResponse,
    ClusteringRequest, ClusteringResponse,
    ChatRequest, ChatResponse
)
from app.services.recommendation_service import RecommendationService
from app.services.prediction_service import PredictionService
from app.services.clustering_service import ClusteringService
from app.services.chat_service import ChatService


# Global variables untuk models
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup"""
    print("Loading models...")
    
    model_dir = Path("trained_models")
    
    # Load SentenceTransformer
    models['transformer'] = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load ML classifiers
    with open(model_dir / "priority_classifier.pkl", "rb") as f:
        models['priority_clf'] = pickle.load(f)
    
    with open(model_dir / "issuetype_classifier.pkl", "rb") as f:
        models['issuetype_clf'] = pickle.load(f)
    
    with open(model_dir / "component_classifier.pkl", "rb") as f:
        models['component_clf'] = pickle.load(f)
    
    with open(model_dir / "component_mlb.pkl", "rb") as f:
        models['component_meta'] = pickle.load(f)
    
    with open(model_dir / "kmeans_model.pkl", "rb") as f:
        models['kmeans'] = pickle.load(f)
    
    # Load embeddings and dataframe
    models['issue_embeddings'] = np.load(model_dir / "issue_embeddings.npy")
    models['df'] = pd.read_pickle(model_dir / "graphs_df.pkl")
    
    # Initialize services
    models['recommendation_service'] = RecommendationService(
        models['transformer'], 
        models['df'], 
        models['issue_embeddings']
    )
    
    models['prediction_service'] = PredictionService(
        models['transformer'],
        models['priority_clf'],
        models['issuetype_clf'],
        models['component_clf'],
        models['component_meta']
    )
    
    models['clustering_service'] = ClusteringService(
        models['transformer'],
        models['kmeans'],
        models['df'],
        models['issue_embeddings']
    )
    
    # Chat service (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        models['chat_service'] = ChatService(
            models['transformer'],
            models['df'],
            models['issue_embeddings']
        )
        print("✅ Chat service enabled (OpenAI API key found)")
    else:
        print("⚠️  Chat service disabled (OPENAI_API_KEY not set)")
    
    print("✅ All models loaded successfully!")
    
    yield
    
    # Cleanup
    models.clear()
    print("Models unloaded")


# Create FastAPI app
app = FastAPI(
    title="Knowledge Hub API",
    description="REST API for method recommendation, ML predictions, clustering, and RAG chatbot based on software issue data",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Knowledge Hub API is running",
        "version": "1.0.0",
        "endpoints": {
            "recommendations": "/api/recommendations",
            "predict_priority": "/api/predict/priority",
            "predict_issuetype": "/api/predict/issuetype",
            "predict_components": "/api/predict/components",
            "cluster": "/api/cluster",
            "chat": "/api/chat"
        }
    }


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get method recommendations based on input features.
    Features akan dikirim dari frontend sebagai array of objects.
    """
    try:
        # Convert Pydantic models to dicts
        features_dict = [feat.model_dump() for feat in request.features]
        
        results = models['recommendation_service'].get_recommendations(
            features_dict,
            top_n=request.top_n,
            blacklists=request.blacklists
        )
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/priority", response_model=PriorityPredictionResponse)
async def predict_priority(request: TextPredictionRequest):
    """Predict priority from summary and description"""
    try:
        result = models['prediction_service'].predict_priority(
            request.summary, 
            request.description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/issuetype", response_model=IssueTypePredictionResponse)
async def predict_issuetype(request: TextPredictionRequest):
    """Predict issue type from summary and description"""
    try:
        result = models['prediction_service'].predict_issuetype(
            request.summary, 
            request.description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/components", response_model=ComponentPredictionResponse)
async def predict_components(request: TextPredictionRequest):
    """Predict components from summary and description (multi-label)"""
    try:
        result = models['prediction_service'].predict_components(
            request.summary, 
            request.description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cluster", response_model=ClusteringResponse)
async def predict_cluster(request: ClusteringRequest):
    """Predict cluster and get similar issues"""
    try:
        result = models['clustering_service'].predict_cluster(
            request.summary,
            request.description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG chatbot endpoint (requires OpenAI API key)"""
    if 'chat_service' not in models:
        raise HTTPException(
            status_code=503, 
            detail="Chat service unavailable. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        result = models['chat_service'].process_query(
            request.query,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
