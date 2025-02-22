# app.py
from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
import config
import utils

app = Flask(__name__)

# Load the saved model at startup
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

# Home route with a simple HTML upload form
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
      <head>
        <title>Car Name Recognition AI</title>
      </head>
      <body>
        <h1>Upload a Car Image for Recognition</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*">
          <br><br>
          <input type="submit" value="Predict">
        </form>
      </body>
    </html>
    '''

# Prediction route to handle uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'})
    
    # Save the uploaded file temporarily
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    # Preprocess the image using the utility function
    img = utils.prepare_image(file_path, config.IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict using the loaded model
    prediction = model.predict(img)
    
    # Create a temporary generator to obtain the mapping of class indices to class names
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    temp_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        config.DATASET_DIR, 
        target_size=config.IMAGE_SIZE, 
        batch_size=1, 
        class_mode='categorical'
    )
    class_indices = temp_gen.class_indices
    inv_class_indices = {v: k for k, v in class_indices.items()}
    
    # Get the predicted class name
    predicted_class = inv_class_indices[np.argmax(prediction)]
    
    # Remove the temporary file
    os.remove(file_path)
    
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

