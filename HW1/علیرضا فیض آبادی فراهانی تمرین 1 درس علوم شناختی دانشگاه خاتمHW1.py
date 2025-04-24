import tempfile
import scipy.io
import zipfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import cv2

# Define paths (replace with your actual paths)
mat_file_path = "C:\\Users\\Lenovo\\Desktop\\S01.mat"  # Path to the S01.mat file
zip_file_path = "C:\\Users\\Lenovo\\Desktop\\Dependencies.zip"  # Path to Dependencies.zip

# load mat data
mat_data = scipy.io.loadmat(mat_file_path)
# Access the data correctly
s01_data = mat_data['S01']

# Extract ET_clean data
et_clean = s01_data['ET_clean']

# Get the relevant data - this is the key change
et_data = et_clean[0, 0][0, 0]
eye_left = et_data[0][:2, :]  # First two rows (x, y) of the left eye

print("Eye-tracking data shape:", eye_left.shape)  # Check the shape

# handle missing values 
eye_left_cleaned =np.nan_to_num(eye_left, nan=np.nanmean(eye_left))

print("shape after handeling nans :", eye_left_cleaned.shape)
print("number of nan remainings :", np.sum(np.isnan(eye_left_cleaned)))

# Image handling with proper error checking
try:
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # Debug: List all files in the zip
        all_files = z.namelist()
        print("Files in zip:", all_files)
        
        # List all files and filter for images (including .tif)
        image_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        if not image_files:
            raise ValueError("No image files found in zip")
            
        # Read first image
        with z.open(image_files[0]) as image_file:
            img_data = image_file.read()
            
        # Open image directly from bytes
        img = Image.open(io.BytesIO(img_data))
        
        # Display image with eye-tracking data overlay
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.scatter(eye_left_cleaned[0], eye_left_cleaned[1], c='red', alpha=0.5, s=1)
        plt.axis('off')
        plt.show()
        
        print("Image dimensions:", img.size)
        
except zipfile.BadZipFile:
    print("Error: The zip file is corrupted or not a valid zip file.")
except FileNotFoundError:
    print(f"Error: The zip file was not found at the specified path: {zip_file_path}")
except Exception as e:
    print(f"Error processing image: {str(e)}")

# we are showing the parts participats more look at

# calculate the image dimensions

img_width , img_height = img.size

# Reverse normalization for gaze estimation
# Assuming the eye coordinates are normalized between 0 and 1
eye_left_x = eye_left_cleaned[0] * img_width
eye_left_y = eye_left_cleaned[1] * img_height

print("Reversed X coordinates:", eye_left_x[:10])  # Print first 10 for example
print("Reversed Y coordinates:", eye_left_y[:10])

# showing the image data

plt.figure(figsize=(10,8))
plt.imshow(img)
plt.scatter(eye_left_x , eye_left_y , c="red" , alpha=0.5 , s=1)
plt.axis('off')
plt.show()

print("image dimensions :", img.size)

# question 3 - Enhanced video generation with heat map and clear gaze visualization
import time

# Video parameters
fps = 120  # Reduced FPS for longer duration
output_video_path = "gaze_tracking_enhanced.avi"

# Convert PIL image to OpenCV format
cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(img_width), int(img_height)))

# Parameters for visualization
trail_length = 60  # Increased trail length
alpha = 0.6  # Increased opacity for better visibility
heatmap_sigma = 30  # Gaussian blur sigma for heatmap
frame_window = 5  # Number of frames to show each position

# Create accumulative heatmap
heatmap = np.zeros((int(img_height), int(img_width)), dtype=np.float32)

# Process frames with slower animation and heat map
for i in range(0, eye_left_x.shape[0], frame_window):
    # Create a fresh copy of the image for this frame
    frame = cv_image.copy()
    
    # Update heatmap for current position
    x_coord = int(np.clip(eye_left_x[i], 0, img_width - 1))
    y_coord = int(np.clip(eye_left_y[i], 0, img_height - 1))
    
    # Add gaussian at current position to heatmap
    y, x = np.ogrid[-y_coord:img_height-y_coord, -x_coord:img_width-x_coord]
    mask = x*x + y*y <= heatmap_sigma*heatmap_sigma
    heatmap[mask] += 1
    
    # Normalize and apply color map to heatmap (changed to RED colormap)
    heatmap_normalized = cv2.GaussianBlur(heatmap, (21, 21), heatmap_sigma)
    heatmap_normalized = (heatmap_normalized - heatmap_normalized.min()) / \
                        (heatmap_normalized.max() - heatmap_normalized.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_HOT)  # Changed to HOT colormap for red emphasis
    
    # Blend heatmap with original frame
    cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0, frame)
    
    # Draw trailing points with fading effect
    for j in range(max(0, i - trail_length), i):
        # Calculate alpha value for trail
        trail_alpha = alpha * (1 - (i - j) / trail_length)
        
        x_prev = int(np.clip(eye_left_x[j], 0, img_width - 1))
        y_prev = int(np.clip(eye_left_y[j], 0, img_height - 1))
        
        # Draw trail points with decreasing size and opacity (changed to red)
        size = int(5 * (1 - (i - j) / trail_length)) + 2
        overlay = frame.copy()
        cv2.circle(overlay, (x_prev, y_prev), size, (0, 0, 255), -1)  # Changed to red
        cv2.addWeighted(overlay, trail_alpha, frame, 1 - trail_alpha, 0, frame)
    
    # Draw current gaze point (changed to bright red)
    cv2.circle(frame, (x_coord, y_coord), 8, (0, 0, 255), -1)  # Changed to red
    
    # Add information overlay
    cv2.putText(frame, f'Time: {i/120:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Frame: {i}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write the same frame multiple times to slow down the animation
    for _ in range(frame_window):
        out.write(frame)
    
    # Show progress
    if i % 100 == 0:
        print(f"Processed frame {i}/{eye_left_x.shape[0]}")

out.release()
print(f"Enhanced video saved to: {output_video_path}")

# Also save the final heatmap as an image (changed to red colormap)
heatmap_final = cv2.applyColorMap((cv2.GaussianBlur(heatmap, (21, 21), heatmap_sigma) * 255).astype(np.uint8), 
                                 cv2.COLORMAP_HOT)  # Changed to HOT colormap
cv2.imwrite("gaze_heatmap.png", heatmap_final)