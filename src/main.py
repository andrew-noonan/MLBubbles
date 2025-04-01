# Entry point of the application
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_mat_dataset(directory_path):
    """
    Load .mat files containing grayscale images and circle annotations.
    
    Args:
        directory_path: Path to directory containing .mat files
        
    Returns:
        images: List of grayscale images
        annotations: List of dictionaries with bubble annotations
    """
    images = []
    annotations = []
    
    # List all .mat files in the directory
    mat_files = [f for f in os.listdir(directory_path) if f.endswith('.mat')]
    
    for mat_file in mat_files:
        # Load .mat file
        mat_path = os.path.join(directory_path, mat_file)
        mat_data = scipy.io.loadmat(mat_path)
        
        # Extract image
        img = mat_data['img']
        
        # Extract circle annotations
        img_circles = mat_data['imgCircles']
        centroids = img_circles['centroid'][0, 0]
        diameters = img_circles['diameterPixels'][0, 0]
        
        # Create annotation list for this image
        img_annotations = []
        for i in range(len(centroids)):
            annotation = {
                'centroidX': float(centroids[i, 0]),
                'centroidY': float(centroids[i, 1]),
                'diameter': float(diameters[i])
            }
            img_annotations.append(annotation)
        
        images.append(img)
        annotations.append(img_annotations)
    
    return images, annotations

# Visualize an image with its annotations to verify
def visualize_sample(images, annotations, index=0):
    plt.figure(figsize=(10, 8))
    plt.imshow(images[index], cmap='gray')
    
    for bubble in annotations[index]:
        x, y, d = bubble['centroidX'], bubble['centroidY'], bubble['diameter']
        circle = plt.Circle((x, y), d/2, fill=False, color='red')
        plt.gca().add_patch(circle)
    
    plt.title(f'Sample Image {index}')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Set path to your .mat files directory
    data_dir = "path/to/mat/files"
    
    # Load dataset
    images, annotations = load_mat_dataset(data_dir)
    
    print(f"Loaded {len(images)} images with annotations")
    
    # Visualize a sample
    if len(images) > 0:
        visualize_sample(images, annotations)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, annotations, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")