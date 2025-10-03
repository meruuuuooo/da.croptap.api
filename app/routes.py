from flask import Blueprint, request, jsonify
from app.services import get_crop_recommendation, get_model_information, retrain_model

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Crop Recommender API is running'
    }), 200

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict the best crop for given conditions
    
    Expected JSON body:
    {
        "soil_type": "Loamy",
        "soil_ph": 6.5,
        "temperature": 25.0,
        "humidity": 80.0,
        "nitrogen": 70.0,
        "phosphorus": 60.0,
        "potassium": 50.0
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        result, status_code = get_crop_recommendation(data)
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Request error: {str(e)}'
        }), 500

@api_bp.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    result, status_code = get_model_information()
    return jsonify(result), status_code

@api_bp.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model (admin endpoint)"""
    # You can add authentication here for security
    result, status_code = retrain_model()
    return jsonify(result), status_code

@api_bp.route('/soil-types', methods=['GET'])
def get_soil_types():
    """Get available soil types"""
    info, _ = get_model_information()
    if 'data' in info:
        return jsonify({
            'success': True,
            'soil_types': info['data'].get('available_soil_types', [])
        }), 200
    return jsonify({
        'success': False,
        'error': 'Model not initialized'
    }), 500

@api_bp.route('/crops', methods=['GET'])
def get_crops():
    """Get available crops"""
    info, _ = get_model_information()
    if 'data' in info:
        return jsonify({
            'success': True,
            'crops': info['data'].get('available_crops', [])
        }), 200
    return jsonify({
        'success': False,
        'error': 'Model not initialized'
    }), 500

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
