from app.models import CropRecommenderModel
import os

# Initialize the model
model = CropRecommenderModel()

def initialize_model():
    """Initialize or load the trained model"""
    try:
        # Try to load existing model
        model.load_model()
        print("Model loaded successfully from disk.")
    except FileNotFoundError:
        print("No trained model found. Training new model...")
        # Train a new model
        csv_path = 'crop_recommender_dataset.csv'
        if os.path.exists(csv_path):
            accuracy = model.train(csv_path)
            print(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

def get_crop_recommendation(data):
    """Get crop recommendation based on input parameters"""
    try:
        soil_type = data.get('soil_type')
        soil_ph = float(data.get('soil_ph'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        nitrogen = float(data.get('nitrogen'))
        phosphorus = float(data.get('phosphorus'))
        potassium = float(data.get('potassium'))
        
        # Validate inputs
        if not soil_type:
            return {'error': 'Soil type is required'}, 400
        
        if not (4.0 <= soil_ph <= 9.0):
            return {'error': 'Soil pH must be between 4.0 and 9.0'}, 400
        
        if not (0 <= temperature <= 50):
            return {'error': 'Temperature must be between 0°C and 50°C'}, 400
        
        if not (0 <= humidity <= 100):
            return {'error': 'Humidity must be between 0% and 100%'}, 400
        
        if nitrogen < 0 or phosphorus < 0 or potassium < 0:
            return {'error': 'NPK values must be non-negative'}, 400
        
        # Get prediction
        result = model.predict(
            soil_type=soil_type,
            soil_ph=soil_ph,
            temperature=temperature,
            humidity=humidity,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium
        )
        
        return {
            'success': True,
            'data': result,
            'input': {
                'soil_type': soil_type,
                'soil_ph': soil_ph,
                'temperature': temperature,
                'humidity': humidity,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium
            }
        }, 200
        
    except ValueError as e:
        return {'error': str(e)}, 400
    except Exception as e:
        return {'error': f'An error occurred: {str(e)}'}, 500

def get_model_information():
    """Get information about the trained model"""
    info = model.get_model_info()
    if info is None:
        return {'error': 'Model not initialized'}, 500
    
    return {
        'success': True,
        'data': info
    }, 200

def retrain_model():
    """Retrain the model with the dataset"""
    try:
        csv_path = 'crop_recommender_dataset.csv'
        if not os.path.exists(csv_path):
            return {'error': 'Dataset file not found'}, 404
        
        accuracy = model.train(csv_path)
        
        return {
            'success': True,
            'message': 'Model retrained successfully',
            'accuracy': accuracy
        }, 200
    except Exception as e:
        return {'error': f'Training failed: {str(e)}'}, 500
