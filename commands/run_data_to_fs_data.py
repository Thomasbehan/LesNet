import os
import random
import re
import shutil

from skinvestigatorai.config.model import ModelConfig


def create_destination_directories():
    if not os.path.exists(ModelConfig.SUPPORT_DIR):
        os.makedirs(ModelConfig.SUPPORT_DIR)
    if not os.path.exists(ModelConfig.QUERY_DIR):
        os.makedirs(ModelConfig.QUERY_DIR)


def sample_images():
    classes = os.listdir(ModelConfig.TRAIN_DIR)
    for class_name in classes:
        class_dir = os.path.join(ModelConfig.TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        pattern = r'^.*augmented_\d+\.jpg$'

        original_images = [img for img in os.listdir(class_dir) if not re.match(pattern, img)]
        augmented_images = [img for img in os.listdir(class_dir) if re.match(pattern, img)]

        random.shuffle(original_images)
        random.shuffle(augmented_images)

        sampled_images = []
        while len(sampled_images) < ModelConfig.FS_NUM_IMAGES_PER_CLASS * 2 and original_images:
            sampled_images.append(original_images.pop())

        while len(sampled_images) < ModelConfig.FS_NUM_IMAGES_PER_CLASS * 2 and augmented_images:
            sampled_images.append(augmented_images.pop())

        for i, img in enumerate(sampled_images):
            source_path = os.path.join(class_dir, img)
            if i < ModelConfig.FS_NUM_IMAGES_PER_CLASS:
                destination_dir = os.path.join(ModelConfig.SUPPORT_DIR, class_name)
            else:
                destination_dir = os.path.join(ModelConfig.QUERY_DIR, class_name)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            destination_path = os.path.join(destination_dir, img)
            shutil.copyfile(source_path, destination_path)


if __name__ == "__main__":
    create_destination_directories()
    sample_images()
    print("Sampling and copying completed.")
