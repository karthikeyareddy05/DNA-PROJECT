import os
import random
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class labels (same order as during training)
class_labels = [
    'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 
    'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 
    'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 
    'papaya', 'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 
    'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_species_demo(image_path):
    """Demo prediction function - returns random results for demonstration"""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Simulate processing time
        import time
        time.sleep(1)
        
        # Return a random prediction for demo purposes
        predicted_class = random.choice(class_labels)
        confidence = round(random.uniform(0.75, 0.95), 2)
        
        logger.info(f"Demo prediction: {predicted_class} (confidence: {confidence:.2f})")
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Error in demo prediction: {e}")
        return None, 0.0

@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index_lightweight.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        logger.info("Received prediction request")
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            logger.info(f"Processing file: {file.filename}")
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make demo prediction
            predicted_class, confidence = predict_species_demo(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            if predicted_class is None:
                return jsonify({'error': 'Prediction failed. Please try a different image.'}), 500
            
            # Format response
            result = {
                'predicted_species': predicted_class.replace('_', ' ').title(),
                'confidence': round(confidence * 100, 2),
                'status': 'success',
                'demo_mode': True,
                'message': 'This is a demo version. For full functionality, the model needs to be deployed with proper resources.'
            }
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'demo_mode': True,
        'message': 'Lightweight demo version running'
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'App is running!', 
        'demo_mode': True,
        'supported_species': len(class_labels)
    })

@app.route('/info')
def info():
    """Information about the app"""
    return jsonify({
        'app_name': 'Plant Species Detection',
        'version': 'Demo 1.0',
        'demo_mode': True,
        'supported_species': class_labels,
        'total_species': len(class_labels),
        'message': 'This is a lightweight demo version. The full model requires more resources than available on the free tier.'
    })

if __name__ == '__main__':
    logger.info("Starting lightweight Flask application...")
    logger.info("Running in demo mode")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
