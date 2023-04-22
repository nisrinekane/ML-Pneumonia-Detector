from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_utils import generate_data_test

model = load_model('best_model.h5')

test_image_directory = 'chest_xray/test'
image_size = (224, 224)
batch_size = 32

# function for preprocessing test images
def preprocess_test_images():
    return ImageDataGenerator(rescale=1./255)

test_datagen = preprocess_test_images()

test_generator = generate_data_test(
    test_datagen,
    test_image_directory,
    image_size,
    batch_size
)

# evaluate model
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
