from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import tempfile
import os
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (model.h5)
model = load_model('model.h5')

# Classification labels (CIFAR-10 classes)
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def home():
    return render_template('index.html')  # Home route to render the upload page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
            print(f"Temp image saved at: {temp_file_path}")

        # Ensure the uploaded file is a valid image and process it
        img = Image.open(temp_file_path)
        img = img.convert('RGB')  # Ensure image is in RGB mode

        # Resize the image to 32x32 (as expected by the model)
        img = img.resize((32, 32))

        # Convert image to numpy array and normalize pixel values
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)

        # Get top 5 predictions
        top_indices = np.argsort(prediction[0])[::-1][:5]

        result = {}
        for i in top_indices:
            result[classification[i]] = round(prediction[0][i] * 100, 2)

        # Process keypoints (SIFT) and add as part of the response
        keypoints_image = process_keypoints(temp_file_path)

        if keypoints_image is None:
            return jsonify({'error': 'Failed to process keypoints'}), 500

        # Convert the image with keypoints to base64 for embedding in the webpage
        keypoints_image_base64 = encode_image_to_base64(keypoints_image)

        # Convert the original image to base64 for rendering in the result page
        original_image_base64 = encode_image_to_base64(cv2.imread(temp_file_path))

        # Remove the temporary file
        os.remove(temp_file_path)

        # Render the result.html template with prediction and images
        return render_template('result.html', 
                               prediction=result, 
                               original_image=original_image_base64, 
                               keypoints_image=keypoints_image_base64)
    
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500


def encode_image_to_base64(image):
    """ Converts image to base64 encoding to display in the browser. """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def process_keypoints(file_path):
    """ Extract SIFT keypoints from the image and return a processed image. """
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            raise ValueError(f"The file {file_path} does not exist.")
        
        # Read the image using OpenCV
        image = cv2.imread(file_path)

        # Check if the image was loaded correctly
        if image is None:
            raise ValueError(f"Failed to read the image from {file_path}. The file might be corrupted or not supported.")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints using SIFT
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # Check if keypoints were detected
        if not keypoints:
            raise ValueError("No keypoints detected in the image.")

        # Draw keypoints on the image
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, image)

        return img_with_keypoints

    except Exception as e:
        print(f"Error processing keypoints: {e}")
        return None


if __name__ == '__main__':
    app.run(debug=True)
