import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = (224, 224)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
try:
    model = load_model("best_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_species(image_path):
    """Predict plant species from image"""
    if model is None:
        return None, 0.0
    
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get predicted class
        predicted_class = class_labels[class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0

@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, confidence = predict_species(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if predicted_class is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
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
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
