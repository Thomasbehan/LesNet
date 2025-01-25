import os
from PIL import Image
from lesnet.config.model import ModelConfig

def validate_images(directory):
    corrupted_files = []
    for dirpath, _, filenames in os.walk(directory):
        for image_file in filenames:
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(dirpath, image_file)
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted_files.append(image_path)
                    print(f"Error with {image_path}: {e}")
    return corrupted_files


# Replace 'path_to_images' with the actual path to your image dataset
dataset_path = ModelConfig.TRAIN_DIR
corrupted_files = validate_images(dataset_path)
for file in corrupted_files:
    os.remove(file)
    print(f"Removed corrupted file: {file}")
print("Corrupted files:", corrupted_files)
