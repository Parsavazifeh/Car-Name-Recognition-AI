import os
import argparse
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config
import utils

def build_model(num_classes):
    if config.MODEL_NAME == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    elif config.MODEL_NAME == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    else:
        raise ValueError("Unsupported MODEL_NAME")
    for layer in base_model.layers[:-70]:
        layer.trainable = False
    for layer in base_model.layers[-70:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def build_hypermodel(hp, num_classes):
    if config.MODEL_NAME == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    elif config.MODEL_NAME == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    else:
        raise ValueError("Unsupported MODEL_NAME")
    unfreeze_layers = hp.Int('unfreeze_layers', min_value=50, max_value=len(base_model.layers), step=10)
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int('dense_units', min_value=512, max_value=2048, step=256), activation='relu')(x)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.0)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='LOG', default=1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=config.AUGMENTATION,
        rotation_range=20 if config.AUGMENTATION else 0,
        zoom_range=0.2 if config.AUGMENTATION else 0
    )
    train_generator = train_datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=config.SEED
    )
    validation_generator = train_datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=config.SEED
    )
    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=config.EPOCHS,
        callbacks=[checkpoint, earlystop, reduce_lr]
    )
    utils.plot_history(history)
    model.save(config.MODEL_SAVE_PATH)
    print("Model saved at:", config.MODEL_SAVE_PATH)

def tune_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=config.AUGMENTATION,
        rotation_range=20 if config.AUGMENTATION else 0,
        zoom_range=0.2 if config.AUGMENTATION else 0
    )
    train_generator = train_datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=config.SEED
    )
    validation_generator = train_datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=config.SEED
    )
    num_classes = len(train_generator.class_indices)
    tuner = kt.RandomSearch(
        lambda hp: build_hypermodel(hp, num_classes),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='kt_dir',
        project_name='car_recognition'
    )
    tuner.search(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=config.EPOCHS,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)]
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=config.EPOCHS,
        callbacks=[ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)]
    )
    utils.plot_history(history)
    model.save(config.MODEL_SAVE_PATH)
    print("Best hyperparameters:", best_hps.values)
    print("Model saved at:", config.MODEL_SAVE_PATH)

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
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--predict', type=str)
    args = parser.parse_args()
    if args.train:
        train_model()
    elif args.tune:
        tune_model()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("No action specified.")