import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image

# defining early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# parameters and paths:
train_csv = 'train.csv'
val_csv = 'val.csv'
test_csv = 'test.csv'

# resize images to same size and set epochs
train_image_directory = 'CXR8/images/images'
val_image_directory = 'CXR8/images/images'
image_size = (224, 224)
batch_size = 32
epochs = 15

# function for data augmentation to diversify data and prevent overfitting
def create_datagen():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

# generate data from csv files
def generate_data(datagen, csv_file, image_folder):
    return datagen.flow_from_dataframe(
        pd.read_csv(csv_file),
        directory=image_folder,
        x_col='id',
        y_col='Label',
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )

# create CNN model using ResNet50 as base model
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# freeze base model layers, train new layers only
for layer in base_model.layers:
    layer.trainable = False

# compile model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# train model 
train_datagen = create_datagen()
val_datagen = create_datagen()
train_generator = generate_data(train_datagen, train_csv, train_image_directory)
val_generator = generate_data(val_datagen, val_csv, val_image_directory)


# fit model 
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=epochs,
    # prevent overfitting
    callbacks=[early_stopping, model_checkpoint] 
)

# save trained model with the best weights using early stopping and save best model checkpoint
model.save('pneumonia_detector.h5')
