import os
import shutil

datasets = ['chest_xray', 'TV-CV-pneumonia-xrays', 'CXR8']
split_folders = ['train', 'val', 'test']
destination_folder = 'merged_dataset'

for split_folder in split_folders:
    print(f"Processing {split_folder}...")
    os.makedirs(os.path.join(destination_folder, split_folder), exist_ok=True)
    for dataset in datasets:
        src_folder = os.path.join(dataset, split_folder)
        for img in os.listdir(src_folder):
            src = os.path.join(src_folder, img)
            if os.path.isfile(src):
                dest = os.path.join(destination_folder, split_folder, img)
                shutil.copy(src, dest)

print("Datasets merged into 'merged_data'.")