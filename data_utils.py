import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_data_train(datagen, csv_file, image_folder, image_size, batch_size):
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

# def generate_data_test(csv_file, image_folder, image_size, batch_size):
#     return ImageDataGenerator().flow_from_dataframe(
#         pd.read_csv(csv_file),
#         directory=image_folder,
#         x_col='id',
#         y_col='Label',
#         target_size=image_size,
#         class_mode='binary',
#         batch_size=batch_size
#     )

# function version for chest x ray folder:
def generate_data_test(datagen, test_image_directory, image_size, batch_size):
    return datagen.flow_from_directory(
        directory=test_image_directory,
        classes=['NORMAL', 'PNEUMONIA'],
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False
    )

