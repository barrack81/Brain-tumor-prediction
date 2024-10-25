from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os
from keras.models import load_model

app = Flask(__name__)

# Make sure the 'uploads' folder exists for storing uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model (adjust the path to where your model is saved)
model_path = 'brain_tumor_model.h5'  # Update this path with your actual model file
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file part is in the request
    if 'file' not in request.files:
        return jsonify({'prediction': 'No file part in the request.'})
    
    file = request.files['file']
    
    # If no file was selected
    if file.filename == '':
        return jsonify({'prediction': 'No file selected for uploading.'})

    try:
        # Save the file to the 'uploads' folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Open the image for processing
        img = Image.open(filepath).convert('RGB')  # Convert to RGB in case it's grayscale
        
        # Resize image to 224x224 for model input
        img = img.resize((224, 224))
        
        # Convert image to numpy array and scale pixel values
        img_array = np.array(img) / 255.0
        
        # Reshape the image to match the input shape (assuming (1, 224, 224, 3))
        img_array = img_array.reshape((1, 224, 224, 3))
        
        # Make prediction using the loaded model
        prediction = model.predict(img_array)
        
        # Assuming binary classification: 0 = No Tumor, 1 = Tumor
        result = 'Tumor detected' if prediction[0][0] > 0.5 else 'No tumor detected'
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'prediction': f"Error processing image: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
