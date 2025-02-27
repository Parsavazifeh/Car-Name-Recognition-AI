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
    # Check for both possible keys for accuracy
    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

