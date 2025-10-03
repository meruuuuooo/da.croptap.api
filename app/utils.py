import os
from datetime import datetime

def ensure_directory_exists(directory):
    """Ensure a directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_request(endpoint, method, data=None):
    """Log API requests for debugging"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {method} {endpoint}"
    if data:
        log_message += f" - Data: {data}"
    print(log_message)

def validate_numeric_input(value, min_val=None, max_val=None):
    """Validate numeric input"""
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False, f"Value must be at least {min_val}"
        if max_val is not None and num > max_val:
            return False, f"Value must be at most {max_val}"
        return True, num
    except (ValueError, TypeError):
        return False, "Invalid numeric value"

def format_response(success, data=None, error=None, message=None):
    """Format API response consistently"""
    response = {'success': success}
    
    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    if message is not None:
        response['message'] = message
    
    return response
