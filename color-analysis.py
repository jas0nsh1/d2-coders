from sklearn.cluster import KMeans
import cv2
import numpy as np

def extract_dominant_colors(image_path, num_colors=3):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    pixels = image.reshape(-1, 3)  # Reshape to a list of pixels
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors


image_path = "township-ad.jpg"  # Replace with your image path
dominant_colors = extract_dominant_colors(image_path)
print("Dominant Colors (RGB):", dominant_colors)