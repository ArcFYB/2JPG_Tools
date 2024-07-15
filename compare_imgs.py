import matplotlib.pyplot as plt
import cv2
import os

def read_image(image_path):
    """Read an image using OpenCV specifically for PNG format."""
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def plot_sr_results_with_roi(original_path, image_paths, roi, titles, save_path=None):
    """
    Plots the original image with ROI and cropped super-resolution results in a 1x4 grid.

    Args:
    - original_path (str): Path to the original image.
    - image_paths (list): List of paths to images to be cropped.
    - roi (tuple): Region of interest defined as (x, y, width, height).
    - titles (list): List of titles for each image.
    - save_path (str, optional): Path to save the image with the ROI.
    """
    # Read the original image
    original_image = read_image(original_path)
    if original_image is None:
        raise FileNotFoundError(f"Cannot read image from {original_path}")

    images = []
    for path in image_paths:
        img = read_image(path)
        if img is None:
            print(f"Warning: Cannot read image from {path}")
        images.append(img)

    x, y, width, height = roi

    # Crop the region of interest
    cropped_images = []
    for img in images:
        if img is not None:
            cropped_img = img[y:y+height, x:x+width]
            if cropped_img.size == 0:
                print(f"Warning: Cropped image from {path} is empty")
            cropped_images.append(cropped_img)
        else:
            cropped_images.append(None)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Display the original image and draw the region of interest
    original_image_with_rect = original_image.copy()
    rect = plt.Rectangle((x, y), width, height, edgecolor='red', facecolor='none', linewidth=2)
    axs[0].imshow(cv2.cvtColor(original_image_with_rect, cv2.COLOR_BGRA2RGBA))
    axs[0].add_patch(rect)
    axs[0].axis('off')
    
    if save_path:
        cv2.rectangle(original_image_with_rect, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.imwrite(save_path, original_image_with_rect)

    # Display the cropped region of interest
    for ax, img, title in zip(axs[1:], cropped_images, titles):
        if img is not None:
            if img.shape[2] == 4:  # Handle PNG with alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[2] == 3:  # Handle regular RGB images
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
original_path = '/home/fiko/Code/DATASET/508_Dataset/TX_Valdation_512/block_31744_8704.jpg'       # Original image path
image_paths = [
    '/home/fiko/Code/DATASET/508_Dataset/TX_Valdation_512/block_0_24576.jpg' ,       # Bicubic
    '/home/fiko/Code/DATASET/508_Dataset/TX_Valdation_512/block_0_24576.jpg' ,  # RCAN
    '/home/fiko/Code/DATASET/508_Dataset/TX_Valdation_512/block_0_24576.jpg'         # Ground Truth
]

roi = (128, 256, 128, 128)  # Replace with your region of interest coordinates
titles = ['Bicubic', 'RCAN', 'Ground Truth']
save_path = './block_0_24576_with_roi.jpg'

plot_sr_results_with_roi(original_path, image_paths, roi, titles, save_path)
