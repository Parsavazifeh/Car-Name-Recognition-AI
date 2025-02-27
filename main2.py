import os
import argparse
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import config
import utils

AUTOTUNE = tf.data.AUTOTUNE

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=config.NUM_CHANNELS)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_image(image, label):
    # Keep only basic augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    return image, label

def prepare_dataset(data_dir, subset='training', batch_size=32, shuffle=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
        validation_split=0.2,
        subset=subset
    )
    if subset == 'training':
        # Unbatch so that augmentations apply to individual images, then re-batch.
        ds = ds.unbatch().map(augment_image, num_parallel_calls=AUTOTUNE).batch(batch_size)
    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds

class WarmUpLearningRateScheduler(Callback):
    def __init__(self, warmup_batches, target_lr):
        super().__init__()
        self.warmup_batches = warmup_batches
        self.target_lr = target_lr
        self.batch_count = 0

    def on_train_batch_begin(self, batch, logs=None):
        lr_var = self.model.optimizer.learning_rate
        if not isinstance(lr_var, tf.Variable):
            lr_var = tf.Variable(lr_var, dtype=tf.float32)
            self.model.optimizer.learning_rate = lr_var
        if self.batch_count < self.warmup_batches:
            warmup_lr = self.target_lr * (self.batch_count + 1) / self.warmup_batches
            self.model.optimizer.learning_rate.assign(warmup_lr)
        self.batch_count += 1

def build_model(num_classes, unfreeze_layers=20, dense_units=1024, dropout_rate=0.6):
    if config.MODEL_NAME == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    elif config.MODEL_NAME == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False,
                           input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    else:
        raise ValueError("Unsupported MODEL_NAME")
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.003))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def build_hypermodel(hp, num_classes):
    if config.MODEL_NAME == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    elif config.MODEL_NAME == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False,
                           input_tensor=Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS)))
    else:
        raise ValueError("Unsupported MODEL_NAME")
    unfreeze_layers = hp.Int('unfreeze_layers', min_value=20, max_value=100, step=10, default=20)
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    dense_units = hp.Int('dense_units', min_value=512, max_value=2048, step=256, default=1024)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.003))(x)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.6, step=0.1, default=0.6)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    lr = hp.Float('learning_rate', min_value=1e-5, max_value=5e-4, sampling='LOG', default=5e-5)
    batch_size = hp.Int('batch_size', min_value=16, max_value=64, step=16, default=config.BATCH_SIZE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    return model

class LRFinderCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_batches, start_lr, end_lr):
        super().__init__()
        self.num_batches = num_batches
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.batch_count = 0
        self.lrs = []
        self.losses = []

    def on_train_batch_begin(self, batch, logs=None):
        pct = self.batch_count / (self.num_batches - 1)
        lr = self.start_lr * (self.end_lr / self.start_lr) ** pct
        self.model.optimizer.learning_rate.assign(lr)
        self.batch_count += 1

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.learning_rate.numpy())
        self.losses.append(logs.get('loss'))

def learning_rate_finder(ds, num_classes, start_lr=1e-5, end_lr=5e-2):
    model = build_model(num_classes, unfreeze_layers=20, dense_units=1024, dropout_rate=0.6)
    num_batches = tf.data.experimental.cardinality(ds).numpy()
    lr_finder = LRFinderCallback(num_batches=num_batches, start_lr=start_lr, end_lr=end_lr)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    model.fit(ds, epochs=1, callbacks=[lr_finder])
    return lr_finder.lrs, lr_finder.losses

def plot_lr_finder(lrs, losses):
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.plot(lrs, losses, marker='o')
    plt.title('Learning Rate Finder')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Training Loss')
    plt.show()

def train_model():
    train_ds = prepare_dataset(config.DATASET_DIR, subset='training', batch_size=config.BATCH_SIZE, shuffle=True)
    val_ds = prepare_dataset(config.DATASET_DIR, subset='validation', batch_size=config.BATCH_SIZE, shuffle=False)
    class_names = tf.keras.preprocessing.image_dataset_from_directory(
        config.DATASET_DIR, validation_split=0.2, subset='training', seed=config.SEED,
        image_size=config.IMAGE_SIZE).class_names
    num_classes = len(class_names)
    model = build_model(num_classes, unfreeze_layers=20, dense_units=1024, dropout_rate=0.6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    warmup = WarmUpLearningRateScheduler(warmup_batches=10, target_lr=5e-5)
    checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=[warmup, checkpoint, earlystop, reduce_lr]
    )
    utils.plot_history(history)
    model.save(config.MODEL_SAVE_PATH)
    print("Model saved at:", config.MODEL_SAVE_PATH)

def tune_model():
    train_ds = prepare_dataset(config.DATASET_DIR, subset='training', batch_size=config.BATCH_SIZE, shuffle=True)
    val_ds = prepare_dataset(config.DATASET_DIR, subset='validation', batch_size=config.BATCH_SIZE, shuffle=False)
    class_names = tf.keras.preprocessing.image_dataset_from_directory(
        config.DATASET_DIR, validation_split=0.2, subset='training', seed=config.SEED,
        image_size=config.IMAGE_SIZE).class_names
    num_classes = len(class_names)
    tuner = kt.RandomSearch(
        lambda hp: build_hypermodel(hp, num_classes),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='kt_dir',
        project_name='car_recognition'
    )
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)]
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
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
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.DATASET_DIR, batch_size=1, image_size=config.IMAGE_SIZE, shuffle=False
    )
    class_names = ds.class_names
    predicted_class = class_names[np.argmax(prediction)]
    print("Predicted class for {}: {}".format(image_path, predicted_class))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--lr_find', action='store_true')
    parser.add_argument('--predict', type=str)
    args = parser.parse_args()
    if args.lr_find:
        ds = prepare_dataset(config.DATASET_DIR, subset='training', batch_size=config.BATCH_SIZE, shuffle=True)
        class_names = tf.keras.preprocessing.image_dataset_from_directory(
            config.DATASET_DIR, validation_split=0.2, subset='training', seed=config.SEED,
            image_size=config.IMAGE_SIZE).class_names
        num_classes = len(class_names)
        lrs, losses = learning_rate_finder(ds, num_classes, start_lr=1e-5, end_lr=5e-2)
        plot_lr_finder(lrs, losses)
    elif args.train:
        train_model()
    elif args.tune:
        tune_model()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("No action specified.")
