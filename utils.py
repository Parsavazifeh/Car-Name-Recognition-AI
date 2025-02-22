# utils.py
import numpy as np
import cv2
import matplotlib.pyplot as plt

def prepare_image(image_path, image_size):
    """
    Load an image from the specified path, resize it to the given dimensions,
    and normalize its pixel values.
    
    Parameters:
      image_path (str): The path to the image file.
      image_size (tuple): The target image size (width, height).
      
    Returns:
      img (ndarray): Preprocessed image array.
    """
    # Load image in BGR format
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load: " + image_path)
    # Convert image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image to the target dimensions
    img = cv2.resize(img, image_size)
    # Normalize pixel values to range [0, 1]
    img = img.astype("float32") / 255.0
    return img

def plot_history(history):
    """
    Plot the training and validation accuracy and loss curves.
    Save the plot to the 'results' directory and display it.
    
    Parameters:
      history: History object returned by model.fit()
    """
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    # Save the plot
    plt.savefig("results/training_history.png")
    plt.show()
