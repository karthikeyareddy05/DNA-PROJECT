import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import logging

# Configure TensorFlow for minimal memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = (224, 224)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None
model_loaded = False

def load_model_safely():
    """Load model with better error handling"""
    global model, model_loaded
    try:
        logger.info("Attempting to load model...")
        if os.path.exists("best_model.h5"):
            # Load model with minimal memory footprint
            model = load_model("best_model.h5", compile=False)
            logger.info("✅ Model loaded successfully!")
            model_loaded = True
            return True
        else:
            logger.error("❌ Model file 'best_model.h5' not found!")
            model_loaded = False
            return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None
        model_loaded = False
        return False

# Try to load model on startup
load_model_safely()

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

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_species(image_path):
    """Predict plant species from image"""
    if not model_loaded or model is None:
        logger.error("Model not loaded")
        return None, 0.0
    
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess image
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None, 0.0
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get predicted class
        predicted_class = class_labels[class_idx]
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.2f})")
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None, 0.0

@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        logger.info("Received prediction request")
        
        # Check if model is loaded
        if not model_loaded or model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not available. Please try again later.'}), 503
        
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
            
            # Make prediction
            predicted_class, confidence = predict_species(filepath)
            
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
                'status': 'success'
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
        'model_loaded': model_loaded,
        'model_file_exists': os.path.exists("best_model.h5")
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({'message': 'App is running!', 'model_loaded': model_loaded})

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Model loaded: {model_loaded}")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
