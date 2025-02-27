import os
import argparse
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import config
import utils
from tensorflow.keras import layers, applications
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import mixed_precision

# Enable mixed precision for better performance
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def build_model(image_size, num_classes):
    # Use EfficientNet with imagenet weights
    base_model = EfficientNetB0(include_top=False, 
                               weights='imagenet', 
                               input_shape=(image_size[0], image_size[1], 3))
    
    # Freeze initial layers
    base_model.trainable = False
    
    # Enhanced head architecture
    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))
    x = applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def create_data_pipeline(dataset_dir, image_size, batch_size):
    # Advanced augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.efficientnet.preprocess_input,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.7,1.3],
        channel_shift_range=50,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    # Oversampling for class imbalance
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42,
        shuffle=True,
        interpolation='bicubic'
    )

    # Calculate class weights
    class_counts = np.bincount(train_generator.classes)
    max_count = np.max(class_counts)
    class_weights = {i: max_count/count for i, count in enumerate(class_counts)}

    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator, class_weights

def train_model():
    # Configuration
    image_size = (380, 380)  # Increased resolution
    batch_size = 32

    # Create data pipeline
    train_gen, val_gen, class_weights = create_data_pipeline(
        config.DATASET_DIR, 
        image_size,
        batch_size
    )
    
    # Build model
    model = build_model(image_size, train_gen.num_classes)
    
    # Phase 1: Train head
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            mode='max',
            restore_best_weights=True
        )
    ]

    # Phase 1 Training
    print("\n=== Training Head ===")
    model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Phase 2: Gradual Unfreezing
    print("\n=== Fine-tuning ===")
    for i in range(len(model.layers[1].layers)-10, 0, -30):  # Unfreeze in stages
        model.layers[1].trainable = True
        for layer in model.layers[1].layers[:i]:
            layer.trainable = False
        
        # Use lower learning rate for earlier layers
        lr = 3e-5 * (1 - i/len(model.layers[1].layers)) + 1e-6
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        model.fit(
            train_gen,
            epochs=5,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks
        )

    # Final training
    model.save(config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

def predict_image(image_path):
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    img = utils.prepare_image(image_path, config.IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    temp_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=1,
        class_mode='categorical'
    )
    class_indices = temp_gen.class_indices
    inv_class_indices = {v: k for k, v in class_indices.items()}
    predicted_class = inv_class_indices[np.argmax(prediction)]
    print("Predicted class for {}: {}".format(image_path, predicted_class))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', type=str)
    args = parser.parse_args()
    if args.train:
        train_model()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("No action specified.")