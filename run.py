from app import create_app
from app.services import initialize_model
import sys
import os

# Create Flask app
app = create_app()

# Initialize the model on startup
try:
    print("Initializing Crop Recommender Model...")
    initialize_model()
    print("Model initialization complete!")
except Exception as e:
    print(f"Error initializing model: {e}")
    sys.exit(1)

if __name__ == '__main__':
    # Run the Flask app
    print("\n" + "="*50)
    print("Starting Crop Recommender API")
    print("="*50)
    print("\nAPI Endpoints:")
    print("  - POST /api/predict          - Get crop recommendation")
    print("  - GET  /api/model-info       - Get model information")
    print("  - GET  /api/soil-types       - Get available soil types")
    print("  - GET  /api/crops            - Get available crops")
    print("  - GET  /api/health           - Health check")
    print("  - POST /api/retrain          - Retrain the model")
    print("\n" + "="*50 + "\n")
    
    # Get port from environment variable (for Render.com) or default to 5000
    port = int(os.getenv('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )
