#!/bin/bash
set -e

# Check if trained models exist
if [ ! -f "/app/trained_models/model_info.json" ]; then
    echo "⚠️  Trained models not found. Running training first..."
    python train_models.py
    echo "✅ Training completed!"
else
    echo "✅ Trained models found. Skipping training."
fi

# Start the application
echo "🚀 Starting Knowledge Hub API..."
exec "$@"
