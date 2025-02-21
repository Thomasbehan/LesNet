import concurrent.futures
import os
import random
import shutil

from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split

from lesnet.config.model import ModelConfig


def augment_image(image_path, output_path):
    """
    Applies random augmentations to an image and saves it with a numbered suffix.
    This version rotates and zooms enough to remove black corners automatically.
    """
    img = Image.open(image_path)

    # Random flips
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Random brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.42, 1.42)
        img = enhancer.enhance(factor)

    # Random contrast
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.42, 1.42)
        img = enhancer.enhance(factor)

    # Random color
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.42, 1.2)
        img = enhancer.enhance(factor)

    # Random zoom (crop and resize back to original size)
    if random.random() > 0.5:
        width, height = img.size
        zoom_factor = random.uniform(0.6, 1.0)
        crop_width, crop_height = int(width * zoom_factor), int(height * zoom_factor)

        # Crop image with high-quality interpolation
        img = ImageOps.fit(img, (crop_width, crop_height), method=Image.Resampling.BICUBIC)
        img = img.resize((width, height), Image.Resampling.BICUBIC)

    img.save(output_path)


def augment_image_task(cls_path, original_image, augmentation_index):
    """
    A helper that performs a single augmentation task for a given class.
    """
    original_image_path = os.path.join(cls_path, original_image)
    base_name, ext = os.path.splitext(original_image)
    augmented_image_name = f"{base_name}_augmented_{augmentation_index}{ext}"
    augmented_image_path = os.path.join(cls_path, augmented_image_name)
    augment_image(original_image_path, augmented_image_path)
    print(f"Augmented {os.path.basename(cls_path)}: created {augmented_image_name}")


def balance_classes(source_dir):
    """
    Balances the dataset by augmenting images in classes until each class has the same number
    of images as the largest class.
    """
    # Get class directories using os.scandir
    classes = [entry for entry in os.scandir(source_dir) if entry.is_dir()]
    class_counts = {}

    # Count original images (exclude any with "_augmented" in the name)
    for cls in classes:
        original_images = []
        with os.scandir(cls.path) as it:
            for entry in it:
                if entry.is_file() and "_augmented" not in entry.name:
                    original_images.append(entry.name)
        class_counts[cls.name] = len(original_images)

    max_count = max(class_counts.values())
    print(f"Largest class size (based on original images): {max_count}")

    # Create augmentation tasks for classes that need more images
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for cls in classes:
            cls_path = cls.path

            # Gather both original and total images in one scan
            original_images = []
            total_images = []
            with os.scandir(cls_path) as it:
                for entry in it:
                    if entry.is_file():
                        total_images.append(entry.name)
                        if "_augmented" not in entry.name:
                            original_images.append(entry.name)

            current_count = len(total_images)
            num_augmentations = max_count - current_count
            if num_augmentations <= 0:
                continue

            for i in range(1, num_augmentations + 1):
                chosen_original = random.choice(original_images)
                tasks.append(
                    executor.submit(augment_image_task, cls_path, chosen_original, i)
                )

        # Process tasks as they complete
        for future in concurrent.futures.as_completed(tasks):
            # Propagate exceptions if any occurred
            future.result()

    print("Dataset balanced to largest class size.")


def copy_file(src_file, dest_file):
    """
    Helper function to copy a file from src to dest.
    """
    shutil.copy(src_file, dest_file)
    print(f"Copied {src_file} to {dest_file}")


def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Splits a dataset into train, validation, and test sets.

    Args:
        source_dir (str): Path to the source dataset directory.
        dest_dir (str): Path to the destination directory.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.

    The remaining data will be used for testing.
    """
    # Create destination directories for each split
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    classes = [entry for entry in os.scandir(source_dir) if entry.is_dir()]
    copy_tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for cls in classes:
            cls_path = cls.path
            images = [entry.name for entry in os.scandir(cls_path) if entry.is_file()]
            if not images:
                continue

            train_files, temp_files = train_test_split(
                images, test_size=(1 - train_ratio), random_state=42
            )
            val_files, test_files = train_test_split(
                temp_files, test_size=(1 - val_ratio / (1 - train_ratio)), random_state=42
            )

            for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
                split_cls_dir = os.path.join(dest_dir, split, cls.name)
                os.makedirs(split_cls_dir, exist_ok=True)
                for file in split_files:
                    src_file = os.path.join(cls_path, file)
                    dest_file = os.path.join(split_cls_dir, file)
                    copy_tasks.append(executor.submit(copy_file, src_file, dest_file))

        # Process copy tasks as they complete
        for future in concurrent.futures.as_completed(copy_tasks):
            future.result()

    print(f"Dataset successfully split into train/val/test sets at {dest_dir}")


# ------------------ Main Execution ------------------

if __name__ == '__main__':
    base_train_dir = os.path.join("..", ModelConfig.TRAIN_DIR)

    # Step 1: Balance classes by augmenting images (multithreaded)
    balance_classes(base_train_dir)

    # Step 2: Split dataset into train/val/test sets (multithreaded)
    split_dataset(base_train_dir, base_train_dir)
