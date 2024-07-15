import cv2
import numpy as np

def histogram_matching(source_img, target_img):
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    source_channels = cv2.split(source_lab)
    target_channels = cv2.split(target_lab)
    
    # Initialize matched channels list
    matched_channels = []
    
    # Perform histogram matching for each channel
    for src, tgt in zip(source_channels, target_channels):
        # Calculate histograms
        src_hist = cv2.calcHist([src], [0], None, [256], [0, 256])
        tgt_hist = cv2.calcHist([tgt], [0], None, [256], [0, 256])
        
        # Normalize histograms
        src_hist_norm = src_hist / np.sum(src_hist)
        tgt_hist_norm = tgt_hist / np.sum(tgt_hist)
        
        # Compute cumulative distribution functions (CDFs)
        src_cdf = np.cumsum(src_hist_norm)
        tgt_cdf = np.cumsum(tgt_hist_norm)
        
        # Initialize lookup table
        lut = np.zeros(256, dtype=np.uint8)
        
        # Match histograms
        for i in range(256):
            lut[i] = np.argmin(np.abs(tgt_cdf - src_cdf[i]))
        
        # Apply lookup table to source channel
        matched_channel = cv2.LUT(src, lut)
        matched_channels.append(matched_channel)
    
    # Merge matched channels
    matched_img_lab = cv2.merge(matched_channels)
    
    # Convert back to BGR color space
    matched_img = cv2.cvtColor(matched_img_lab, cv2.COLOR_LAB2BGR)
    
    return matched_img

# Load source and target images
source_img = cv2.imread('/home/fiko/Code/Super_Resolution/ddim-diffusion-super-resolution/experiments/sr_Alsat_240305_090705/results/0_2_sr.png')
target_img = cv2.imread('/home/fiko/Code/Super_Resolution/ddim-diffusion-super-resolution/experiments/sr_Alsat_240305_090705/results/0_2_hr.png')

# Perform histogram matching
matched_img = histogram_matching(source_img, target_img)

# Display results
cv2.imshow('Source Image', source_img)
cv2.imshow('Target Image', target_img)
cv2.imshow('Matched Image', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
