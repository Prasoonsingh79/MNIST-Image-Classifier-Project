import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import io
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
model = None
def load_model_from_disk():
    global model
    model_paths = [
        'models/mnist_model_final.keras',
        'models/mnist_model_final.h5',
        'models/mnist_model.h5'
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                print(f"Attempting to load model from {model_path}...")
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
                print(f"Successfully loaded model from {model_path}")
                return
            else:
                print(f"Model not found at {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
    
    print("\nERROR: Failed to load any model. Please run train_mnist.py first to train a new model.")
    print("Make sure you have one of these files in the models/ directory:")
    print("- mnist_model_final.keras")
    print("- mnist_model_final.h5")
    print("- mnist_model.h5")
    model = None

# Load model when starting the app
load_model_from_disk()

def preprocess_image(img_data):
    try:
        # Convert base64 image data to numpy array
        header, encoded = img_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        
        # Convert to PIL Image and ensure it's in grayscale
        img = Image.open(io.BytesIO(binary_data)).convert('L')
        
        # Convert to numpy array first, then invert
        img_array = 255 - np.array(img, dtype='float32')
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Normalize to [0, 1] range
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        # Resize to 28x28 using LANCZOS resampling for better quality
        if img_array.shape[0] != 28 or img_array.shape[1] != 28:
            img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Simple thresholding to make digit more visible
        _, img_array = cv2.threshold(img_array, 0.5, 1.0, cv2.THRESH_BINARY)
        
        # Find the bounding box of the digit
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        
        if True in rows and True in cols:  # Only if digit is detected
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Calculate the center of mass
            r_center = (rmin + rmax) // 2
            c_center = (cmin + cmax) // 2
            
            # Calculate the size of the digit
            digit_height = rmax - rmin
            digit_width = cmax - cmin
            
            # Calculate the size of the square that will contain the digit
            size = max(digit_height, digit_width) + 4  # Add some padding
            size = min(size, 28)  # Make sure it's not larger than the image
            
            # Calculate the region to extract
            r_start = max(0, r_center - size//2)
            c_start = max(0, c_center - size//2)
            r_end = min(28, r_start + size)
            c_end = min(28, c_start + size)
            
            # Extract the digit region
            digit_region = img_array[r_start:r_end, c_start:c_end]
            
            # Resize to 20x20 while maintaining aspect ratio
            if digit_region.size > 0:  # Check if we have a valid region
                digit_region = cv2.resize(digit_region, (20, 20), interpolation=cv2.INTER_AREA)
                
                # Create a new 28x28 image with the digit centered
                img_array = np.zeros((28, 28), dtype='float32')
                start_row = (28 - 20) // 2
                start_col = (28 - 20) // 2
                img_array[start_row:start_row+20, start_col:start_col+20] = digit_region
        
        # Ensure the image is in [0, 1] range
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
            
        # Apply slight blur for smoother edges
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
        
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        raise
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please try again later.'}), 500
    
    try:
        # Get the image data from the request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Check if the canvas is empty (all white)
        if data['image'].endswith('A=='):  # This is a base64 encoded blank white image
            return jsonify({'success': False, 'error': 'Please draw a digit first'}), 400
        
        # Preprocess the image
        img_array = preprocess_image(data['image'])
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]  # Added verbose=0 to suppress TF output
        
        # Convert predictions to a more readable format
        prediction_results = [
            {'digit': int(i), 'probability': float(pred)}
            for i, pred in enumerate(predictions)
        ]
        
        # Sort predictions by probability (descending) and get top 3
        top_predictions = sorted(
            prediction_results, 
            key=lambda x: x['probability'], 
            reverse=True
        )[:3]
        
        # Get the top prediction
        top_prediction = top_predictions[0]
        
        return jsonify({
            'success': True,
            'predictions': top_predictions,
            'top_prediction': top_prediction
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
