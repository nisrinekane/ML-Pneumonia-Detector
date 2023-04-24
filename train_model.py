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
from data_utils import generate_data_train
from PIL import Image
from tensorflow.keras.layers import Dropout

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
def create_datagen(rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0, shear_range=0, horizontal_flip=False, fill_mode="nearest"):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

train_datagen = create_datagen(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)
val_datagen = create_datagen()

# create CNN model using ResNet50 as base model
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# freeze base model layers, train new layers only
# for layer in base_model.layers:
#     layer.trainable = False
# unfreeze top layers 
for layer in base_model.layers[-10:]:
    layer.trainable = True

# compile model with Adam optimizer and binary crossentropy loss
# model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# compile model with lower learning rate
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# train model 
train_generator = generate_data_train(train_datagen, train_csv, train_image_directory, image_size, batch_size)
val_generator = generate_data_train(val_datagen, val_csv, val_image_directory, image_size, batch_size)

# fit model 
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=5,
    # prevent overfitting
    callbacks=[early_stopping, model_checkpoint]
)

# save trained model with the best weights using early stopping and save best model checkpoint
model.save('pneumonia_detector.h5')
