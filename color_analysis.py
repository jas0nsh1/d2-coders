from sklearn.cluster import KMeans
import cv2
import numpy as np

def dominant_RGB(image_path, num_colors=3):
    # Load image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    pixels = image.reshape(-1, 3)  # Reshape to a list of pixels
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # Filter out black or very dark grey colors (where all RGB values are less than 10)
    colors = [color for color in colors if not (color[0] < 10 and color[1] < 10 and color[2] < 10)]

    # If there are not enough colors (e.g., all were black/grey), re-run KMeans with fewer clusters
    if len(colors) < num_colors:
        kmeans = KMeans(n_clusters=num_colors - len(colors))
        kmeans.fit(pixels)
        additional_colors = kmeans.cluster_centers_.astype(int)
        colors.extend([color for color in additional_colors if not (color[0] < 10 and color[1] < 10 and color[2] < 10)])

    # Clip and brighten the remaining colors
    colors = np.clip(np.array(colors) * 1.15, 0, 255)

    return colors.astype(int)


image_path = "t-l.png"  # Replace with your image path
dominant_colors = dominant_RGB(image_path)
print("Dominant Colors (RGB):", dominant_colors)