import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_continuous_segments(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Threshold the image to binary
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    return num_labels, labels_im

# Example usage:
# Create a simple binary image with some continuous segments
binary_image = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0]
], dtype=np.uint8) * 255

num_labels, labels_im = detect_continuous_segments(binary_image)

print(f"Number of continuous segments: {num_labels - 1}")  # Subtract 1 for the background

# Display the original and labeled images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Binary Image")
plt.imshow(binary_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Labeled Image")
plt.imshow(labels_im, cmap='nipy_spectral')
plt.show()
