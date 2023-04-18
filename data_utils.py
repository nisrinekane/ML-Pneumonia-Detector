import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_data_train(csv_file, image_folder, image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    return train_datagen.flow_from_dataframe(
        pd.read_csv(csv_file),
        directory=image_folder,
        x_col='id',
        y_col='Label',
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )

def generate_data_test(csv_file, image_folder, image_size, batch_size):
    return ImageDataGenerator().flow_from_dataframe(
        pd.read_csv(csv_file),
        directory=image_folder,
        x_col='id',
        y_col='Label',
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size
    )
