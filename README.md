# Car Name Recognition AI

A simple AI application that identifies the make and model of a car from a provided image. This project leverages transfer learning with a pre-trained convolutional neural network (CNN) to classify car images.

## Overview

This project demonstrates how to:
- Gather and preprocess a car image dataset.
- Build a deep learning model using a pre-trained CNN (e.g., ResNet50) with custom classification layers.
- Train and evaluate the model on labeled car images.
- Deploy a simple web interface (using Flask) to upload a photo and receive a prediction for the car's make and model.

## Features

- **Image Classification:** Recognize car make and model from images.
- **Transfer Learning:** Utilize a pre-trained model to reduce training time.
- **Web Deployment:** Simple Flask app to test predictions.
- **Data Augmentation:** Optional techniques to improve model generalization.

## Requirements

- **Python:** 3.7+
- **Deep Learning Framework:** TensorFlow 2.x (or PyTorch if preferred)
- **Other Libraries:**
  - OpenCV
  - NumPy
  - Matplotlib
  - Flask
  - scikit-learn
- **Dataset:** A collection of car images organized by class. For example:
  ```
  dataset/
    Honda_Civic/
      img1.jpg
      img2.jpg
    Ford_Focus/
      img1.jpg
      img2.jpg
  ```
- **Hardware:** GPU recommended for faster training, though CPU training is supported.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/car-name-recognition-ai.git
   cd car-name-recognition-ai
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
car-name-recognition-ai/
├── app.py             # Flask web application to serve predictions
├── config.py          # Configuration variables (paths, hyperparameters, etc.)
├── main.py            # Script to preprocess data, train, and evaluate the model
├── utils.py           # Utility functions for image preprocessing and prediction
├── requirements.txt   # List of Python dependencies
├── README.md          # This file
└── dataset/           # Directory containing car images organized by class
```

## Data Organization

Organize your dataset as follows:
```
dataset/
  Honda_Civic/
    image1.jpg
    image2.jpg
  Ford_Focus/
    image1.jpg
    image2.jpg
  ... (other car classes)
```
Each subfolder represents a unique car class (make and model). Ensure you have a balanced set of images for each class.

## Preprocessing

Before training, preprocess the images:
- Resize all images to a consistent shape (e.g., 224x224).
- Normalize pixel values (scale between 0 and 1).
- Optionally, apply data augmentation (flips, rotations, etc.).

A helper function is provided in `utils.py` for image loading and preprocessing.

## Training the Model

1. **Update Hyperparameters:**  
   Adjust the parameters in `config.py` (e.g., epochs, batch size, learning rate).

2. **Run the Training Script:**
   ```bash
   python main.py --train
   ```
   This script will:
   - Load and preprocess the dataset.
   - Build the model using transfer learning.
   - Train and validate the model.
   - Save the trained model to disk.

## Testing the Model

To test a single image prediction, run:
```bash
python main.py --predict path/to/test_image.jpg
```
The script will preprocess the image, load the trained model, and output the predicted car make and model.

## Running the Web Application

A simple Flask app is provided for interactive testing:
1. **Start the Flask Server:**
   ```bash
   python app.py
   ```
2. **Access the Web Interface:**  
   Open your browser and navigate to `http://127.0.0.1:5000`. Use the provided form to upload an image and see the prediction.

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Pre-trained models provided by [Keras Applications](https://keras.io/api/applications/).
- Dataset sources such as the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) or similar.
- Inspiration from various online tutorials and GitHub projects in the computer vision domain.