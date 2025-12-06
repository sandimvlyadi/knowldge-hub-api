# Knowledge Hub API

REST API untuk method recommendation, ML predictions, clustering, dan RAG chatbot berdasarkan data software issues (Jira tickets).

## 🚀 Features

- **Method Recommendations**: Semantic similarity search untuk merekomendasikan methods berdasarkan features
- **Priority Prediction**: ML classifier untuk memprediksi priority issue
- **Issue Type Prediction**: ML classifier untuk memprediksi tipe issue
- **Component Prediction**: Multi-label classifier untuk memprediksi components
- **Clustering**: Grouping issue berdasarkan similarity semantik
- **RAG Chatbot**: AI assistant dengan knowledge base dari historical issues (memerlukan OpenAI API key)

## 📋 Prerequisites

- Python 3.9+
- OpenAI API Key (opsional, hanya untuk chatbot)

## 🛠️ Installation

1. **Clone repository**

```bash
cd knowledge-hub-api
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Setup environment variables (opsional)**

Jika ingin menggunakan chatbot feature, buat file `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 📊 Data Preparation

Pastikan Anda memiliki file data berikut di root directory:
- `graphs.json` - Historical issue data untuk training
- `features.json` - Contoh format input features (untuk referensi)

## 🏋️ Training Models

Sebelum menjalankan API, Anda harus train models terlebih dahulu:

```bash
python train_models.py
```

Script ini akan:
1. Load data dari `graphs.json`
2. Generate embeddings menggunakan SentenceTransformer
3. Train 3 ML classifiers (Priority, IssueType, Component)
4. Train KMeans clustering model
5. Simpan semua models di folder `trained_models/`

Output:
```
trained_models/
├── priority_classifier.pkl
├── issuetype_classifier.pkl
├── component_classifier.pkl
├── component_mlb.pkl
├── kmeans_model.pkl
├── issue_embeddings.npy
├── graphs_df.pkl
└── model_info.json
```

## 🚀 Running the API

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Atau jalankan langsung:

```bash
python -m app.main
```

API akan berjalan di: `http://localhost:8000`

## 📖 API Documentation

Setelah API berjalan, akses dokumentasi interaktif:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### 1. Health Check
```http
GET /
```

### 2. Method Recommendations
```http
POST /api/recommendations
```

**Request Body:**
```json
{
  "features": [
    {
      "key": "FEAT-1",
      "summary": "Add stop goal to maven plugin",
      "description": "This would be useful for stopping ActiveMQ in build processes",
      "components": [],
      "project": "ActiveMQ",
      "issuetype": "New Feature",
      "priority": "Major",
      "status": "Open",
      "reporter": "John Doe",
      "methods": []
    }
  ],
  "top_n": 5,
  "blacklists": ["java.io", "java.util"]
}
```

**Response:**
```json
{
  "results": [
    {
      "feature_key": "FEAT-1",
      "recommendations": [
        {
          "key": "AMQ-3452",
          "score": 0.87,
          "methods": ["org.apache.activemq.broker.BrokerService", "..."]
        }
      ],
      "unique_methods": ["org.apache.activemq.broker.BrokerService", "..."]
    }
  ]
}
```

### 3. Predict Priority
```http
POST /api/predict/priority
```

**Request Body:**
```json
{
  "summary": "Database connection timeout",
  "description": "Application fails to connect to database after 30 seconds"
}
```

**Response:**
```json
{
  "predicted_priority": "Major",
  "confidence": 0.85
}
```

### 4. Predict Issue Type
```http
POST /api/predict/issuetype
```

**Request Body:**
```json
{
  "summary": "Add new API endpoint",
  "description": "Need to expose user statistics via REST API"
}
```

**Response:**
```json
{
  "predicted_issuetype": "New Feature",
  "confidence": 0.92
}
```

### 5. Predict Components
```http
POST /api/predict/components
```

**Request Body:**
```json
{
  "summary": "Fix memory leak in cache",
  "description": "Cache grows indefinitely causing OOM"
}
```

**Response:**
```json
{
  "predicted_components": ["Cache", "Memory Management"],
  "confidence_scores": {
    "Cache": 0.89,
    "Memory Management": 0.76
  }
}
```

### 6. Clustering
```http
POST /api/cluster
```

**Request Body:**
```json
{
  "summary": "Performance issue in search",
  "description": "Search queries take more than 5 seconds"
}
```

**Response:**
```json
{
  "cluster_id": 2,
  "similar_issues": [
    {
      "key": "PROJ-123",
      "summary": "Slow database queries",
      "priority": "High",
      "issuetype": "Bug"
    }
  ]
}
```

### 7. Chat (RAG)
```http
POST /api/chat
```

**Request Body:**
```json
{
  "query": "How to fix ActiveMQ connection issues?",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "Based on historical issues, ActiveMQ connection issues can be resolved by...",
  "sources": [
    {
      "key": "AMQ-3189",
      "summary": "Connection timeout fix",
      "score": 0.91
    }
  ]
}
```

## 🏗️ Project Structure

```
knowledge-hub-api/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── services/
│   │   ├── recommendation_service.py
│   │   ├── prediction_service.py
│   │   ├── clustering_service.py
│   │   └── chat_service.py
│   └── utils/
│       └── data_loader.py
├── trained_models/                # Model artifacts (generated)
├── train_models.py                # Training script
├── graphs.json                    # Training data
├── features.json                  # Example feature input
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Model Configuration
Edit `train_models.py` untuk mengubah:
- `n_clusters`: Jumlah cluster (default: 5)
- `min_count`: Minimum data per component (default: 5)
- SentenceTransformer model: default `all-MiniLM-L6-v2`

### API Configuration
Edit `app/main.py` untuk:
- CORS settings
- Host/port configuration
- Add custom middleware

## 📝 Frontend Integration

Dari frontend, kirim data features dalam format:

```javascript
const features = [
  {
    key: "FEAT-1",
    summary: "Feature summary",
    description: "Feature description",
    components: [],
    project: "Project Name",
    issuetype: "New Feature",
    priority: "Major",
    status: "Open",
    reporter: "Reporter Name",
    methods: []
  }
];

const response = await fetch('http://localhost:8000/api/recommendations', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    features: features,
    top_n: 5,
    blacklists: ["java.io", "java.util"]
  })
});

const data = await response.json();
console.log(data.results);
```

## 🐛 Troubleshooting

**Error: Models not found**
- Jalankan `python train_models.py` terlebih dahulu

**Error: OPENAI_API_KEY not set**
- Chat endpoint memerlukan OpenAI API key
- Set environment variable atau endpoint ini akan disabled

**Error: Out of memory**
- Kurangi `top_n` di request
- Gunakan server dengan RAM lebih besar
- Reduce batch size saat training

## 📦 Deployment

### Docker Deployment

#### 1. Build dan Run dengan Docker Compose (Recommended)

**First time setup (dengan training):**
```bash
# Train models terlebih dahulu
docker-compose --profile training up trainer

# Start API service
docker-compose up -d
```

**Jika models sudah ada:**
```bash
# Langsung start API
docker-compose up -d
```

**Stop service:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f api
```

#### 2. Build dan Run dengan Docker manual

**Build image:**
```bash
docker build -t knowledge-hub-api .
```

**Train models (first time):**
```bash
docker run --rm -v $(pwd)/trained_models:/app/trained_models knowledge-hub-api python train_models.py
```

**Run API:**
```bash
docker run -d \
  --name knowledge-hub-api \
  -p 8000:8000 \
  -v $(pwd)/trained_models:/app/trained_models \
  -e OPENAI_API_KEY="your_key_here" \
  knowledge-hub-api
```

**View logs:**
```bash
docker logs -f knowledge-hub-api
```

**Stop container:**
```bash
docker stop knowledge-hub-api
docker rm knowledge-hub-api
```

### Cloud Deployment

Untuk production deployment:

1. Set environment variables
2. Use production ASGI server (Gunicorn + Uvicorn)
3. Setup reverse proxy (Nginx)
4. Enable SSL/TLS
5. Implement rate limiting
6. Setup logging & monitoring

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License

## 👥 Authors

Knowledge Hub Team

## 🙏 Acknowledgments

- SentenceTransformers
- FastAPI
- Scikit-learn
- OpenAI
