from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_utils import generate_data_test


model = load_model('best_model.h5')

test_image_directory = 'CXR8/images/images'
image_size = (224, 224)
batch_size = 32
test_csv = 'test.csv'

test_generator = generate_data_test(
    test_csv,
    test_image_directory,
    image_size,
    batch_size
)

# evaluate model
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
