import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.io as sio
import os
import multiprocessing
from functools import partial
from skimage import exposure
from tqdm import tqdm

def process_single_image(data_item, return_visualization=False):
    """Process a single image with normalization and auto-contrast"""
    # Extract image
    img = np.array(data_item.image, dtype=np.float32)
    
    # Apply CLAHE for auto-contrast
    img_uint8 = np.uint8(img)
    clahe = exposure.equalize_adapthist(img_uint8, clip_limit=0.04)
    
    # Normalize to 0-1 range
    normalized_img = clahe / np.max(clahe)
    
    # Extract bubble annotations
    x_coords = np.array(data_item.X).flatten()
    y_coords = np.array(data_item.Y).flatten()
    diameters = np.array(data_item.dia).flatten()
    
    # Create annotation list
    annotations = [
        {'centroidX': float(x), 'centroidY': float(y), 'diameter': float(d)}
        for x, y, d in zip(x_coords, y_coords, diameters)
    ]
    
    return normalized_img, annotations

def parallel_process_data(training_data, num_processes=None):
    """Process all images in parallel"""
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Processing {len(training_data)} images using {num_processes} processes...")
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process images in parallel
        results = list(tqdm(
            pool.imap(process_single_image, training_data),
            total=len(training_data)
        ))
    
    # Separate results into images and annotations
    processed_images, annotations = zip(*results)
    
    return list(processed_images), list(annotations)

def main():
    mat_file_path = r"C:\Users\anoon\OneDrive - Vanderbilt\Masters Research\MATLAB\MachineLearning Bubbles\trainingDataCompiled.mat"
    
    # Load data
    print("Loading data...")
    mat_data = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    training_data = mat_data['trainingData']
    
    # Process all images in parallel
    processed_images, annotations = parallel_process_data(training_data)
    print(f"Processed {len(processed_images)} images")
    
    # Optional: visualize a sample
    idx = 2  # Change to view different images
    plt.figure(figsize=(10, 8))
    plt.imshow(processed_images[idx], cmap='gray')
    
    # Add circles for bubbles
    for annot in annotations[idx]:
        x, y, d = annot['centroidX'], annot['centroidY'], annot['diameter']
        circle = Circle((x, y), d/2, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(circle)
    
    plt.title(f"Processed Image {idx+1} with {len(annotations[idx])} Bubbles")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()