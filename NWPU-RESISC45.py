import os
import random
import shutil

def select_random_images(source_dir, target_dir, percentage=0.1):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate through each subfolder in the source directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        
        if os.path.isdir(subdir_path):
            # List all jpg files in the subfolder
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')]
            # Calculate the number of images to select
            num_images_to_select = max(1, int(len(images) * percentage))
            # Randomly select images
            selected_images = random.sample(images, num_images_to_select)
            
            # Create the corresponding subfolder in the target directory
            target_subdir_path = os.path.join(target_dir, subdir)
            os.makedirs(target_subdir_path, exist_ok=True)
            
            # Copy the selected images to the target subfolder
            for image in selected_images:
                source_image_path = os.path.join(subdir_path, image)
                target_image_path = os.path.join(target_subdir_path, image)
                shutil.copy(source_image_path, target_image_path)
            print(f"Copied {num_images_to_select} images from {subdir} to {target_subdir_path}")

source_folder = '/home/fiko/Code/DATASET/NWPU-RESISC45'
target_folder = '/home/fiko/Code/DATASET/NWPU-RESISC45_0.1'

select_random_images(source_folder, target_folder)
