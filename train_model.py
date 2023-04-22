import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_utils import generate_data_train, generate_data_test
from sklearn.model_selection import train_test_split

# Read and preprocess data
data = pd.read_csv('CXR8/LongTailCXR/nih-cxr-lt_single-label_train.csv')
data = data[(data['No Finding'] == 1) | (data['Pneumonia'] == 1)]
data['Label'] = data.apply(lambda row: 'No Finding' if row['No Finding'] == 1 else 'Pneumonia', axis=1)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Label'])
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42, stratify=train_data['Label'])
train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)
test_data.to_csv('test.csv', index=False)

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
epochs = 10

# function for data augmentation to diversify data and prevent overfitting
def create_datagen():
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

# create CNN model using MobileNetV2 as base model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # Add dropout layer
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

# update the function calls to generate_data_train for both training and validation generators
train_generator = generate_data_train(train_datagen, train_csv, train_image_directory, image_size, batch_size)
val_generator = generate_data_train(val_datagen, val_csv, val_image_directory, image_size, batch_size)

# fit model 
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
