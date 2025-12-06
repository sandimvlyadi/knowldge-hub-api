from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureInput(BaseModel):
    """Input schema untuk feature yang akan dicari rekomendasinya"""
    key: str
    summary: str
    description: str
    components: Optional[List[str]] = []
    project: Optional[str] = ""
    issuetype: Optional[str] = ""
    priority: Optional[str] = ""
    status: Optional[str] = ""
    reporter: Optional[str] = ""
    methods: Optional[List[str]] = []


class RecommendationRequest(BaseModel):
    """Request untuk mencari rekomendasi berdasarkan features"""
    features: List[FeatureInput]
    top_n: int = Field(default=5, ge=1, le=20, description="Number of top recommendations per feature")
    blacklists: Optional[List[str]] = Field(
        default=["java.io", "java.util", "java.lang", "org.apache.maven", 
                 "org.junit", "org.slf4j", "org.springframework", "java.net"],
        description="List of method prefixes to exclude from results"
    )


class MethodRecommendation(BaseModel):
    """Single method recommendation"""
    key: str
    score: float
    methods: List[str]


class FeatureRecommendation(BaseModel):
    """Recommendations for a single feature"""
    feature_key: str
    recommendations: List[MethodRecommendation]
    unique_methods: List[str]


class RecommendationResponse(BaseModel):
    """Response containing all recommendations"""
    results: List[FeatureRecommendation]


class TextPredictionRequest(BaseModel):
    """Request untuk prediksi berdasarkan text"""
    summary: str
    description: str


class PriorityPredictionResponse(BaseModel):
    """Response untuk prediksi priority"""
    predicted_priority: str
    confidence: float


class IssueTypePredictionResponse(BaseModel):
    """Response untuk prediksi issue type"""
    predicted_issuetype: str
    confidence: float


class ComponentPredictionResponse(BaseModel):
    """Response untuk prediksi components"""
    predicted_components: List[str]
    confidence_scores: dict


class ClusteringRequest(BaseModel):
    """Request untuk clustering"""
    summary: str
    description: str


class ClusteringResponse(BaseModel):
    """Response untuk clustering"""
    cluster_id: int
    similar_issues: List[dict]


class ChatRequest(BaseModel):
    """Request untuk RAG chatbot"""
    query: str
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context documents to retrieve")


class ChatResponse(BaseModel):
    """Response dari chatbot"""
    answer: str
    sources: List[dict]
