import os
import tarfile

source_folder = 'CXR8/images'
destination_folder = 'CXR8/all_images'

os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith('.tar.gz'):
        file_path = os.path.join(source_folder, filename)
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=destination_folder)

print("Images extracted to 'CXR8/all_images'")
