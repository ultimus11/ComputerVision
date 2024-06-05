import cv2
import numpy as np

# Load an example image
image = cv2.imread('example.jpg')

# 1. Grayscale Conversion
"""
Converting images from RGB to grayscale simplifies the data by reducing the number of color channels from three to one.
"""
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Histogram Equalization
"""
Histogram equalization enhances the contrast of images by spreading out the most frequent intensity values.
"""
equalized_image = cv2.equalizeHist(gray_image)

# 3. Gaussian Blur
"""
Applying a Gaussian blur smoothens an image by averaging the pixel values with their neighbors.
"""
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 4. Median Filtering
"""
Median filtering reduces salt-and-pepper noise in an image.
"""
median_filtered_image = cv2.medianBlur(image, 5)

# 5. Bilateral Filtering
"""
Bilateral filtering smooths images while preserving edges by averaging pixels based on both their spatial closeness and intensity similarity.
"""
bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)

# 6. Image Resizing
"""
Image resizing adjusts the dimensions of an image.
"""
resized_image = cv2.resize(image, (100, 100))  # Example size

# 7. Normalization
"""
Normalization adjusts the pixel intensity values to a standard range.
"""
normalized_image = image / 255.0

# 8. Edge Detection
"""
Edge detection algorithms like Canny identify the boundaries within an image.
"""
edges = cv2.Canny(image, 100, 200)

# 9. Thresholding
"""
Thresholding converts grayscale images into binary images.
"""
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 10. Morphological Operations
"""
Erosion and dilation are fundamental morphological operations.
"""
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(binary_image, kernel, iterations=1)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# 11. Contour Detection
"""
Contour detection identifies the outlines of objects within an image.
"""
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# 12. Color Space Conversion
"""
Converting images to different color spaces can highlight certain features.
"""
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 13. Image Augmentation
"""
Image augmentation techniques generate variations of an image.
"""
augmented_image = cv2.flip(image, 1)  # Horizontal flip

# 14. Adaptive Thresholding
"""
Adaptive thresholding calculates thresholds for smaller regions of the image.
"""
adaptive_thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 15. Image Pyramids
"""
Image pyramids create multi-scale representations of an image.
"""
gaussian_pyramid = [image]
for i in range(6):
    image = cv2.pyrDown(image)
    gaussian_pyramid.append(image)

# 16. Background Subtraction
"""
Background subtraction isolates foreground objects from the background.
"""
fgbg = cv2.createBackgroundSubtractorMOG2()
foreground_mask = fgbg.apply(image)

# 17. Homography
"""
Homography transformations correct perspective distortions.
"""
# Assuming src_pts and dst_pts are defined
# src_pts = ...
# dst_pts = ...
# h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 18. Image De-noising
"""
Image de-noising techniques aim to remove noise while preserving important image details.
"""
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 19. CLAHE (Contrast Limited Adaptive Histogram Equalization)
"""
CLAHE improves local contrast and enhances the definition of edges in each region of an image.
"""
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# 20. Image Sharpening
"""
Image sharpening enhances edges and fine details in an image.
"""
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, kernel)

# 21. Log Transformation
"""
Log transformation enhances the details in darker regions of the image.
"""
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))
log_image = np.array(log_image, dtype=np.uint8)

# 22. Gamma Correction
"""
Gamma correction adjusts the brightness of an image.
"""
gamma = 2.2
gamma_corrected_image = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# 23. Fourier Transform
"""
The Fourier Transform is used to analyze the frequency components of an image.
"""
dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 24. Hough Transform
"""
Hough Transform is a technique to detect lines, circles, and other parametric shapes in an image.
"""
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# 25. Image Binarization with Otsu's Method
"""
Otsu's method automatically determines the optimal threshold value for binarization.
"""
_, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 26. Affine Transformation
"""
Affine transformation includes scaling, rotating, and translating the image.
"""
rows, cols = image.shape[:2]
matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
affine_transformed_image = cv2.warpAffine(image, matrix, (cols, rows))

# 27. Perspective Transformation
"""
Perspective transformation changes the viewing perspective of an image.
"""
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [200, 200]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective_image = cv2.warpPerspective(image, matrix, (cols, rows))

# 28. Watershed Algorithm
"""
The Watershed algorithm is used for image segmentation, treating the grayscale image like a topographic map.
"""
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

# 29. Sobel Filter
"""
The Sobel filter is used for edge detection, highlighting regions of high spatial frequency.
"""
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
sobel_image = cv2.sqrt(cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0))

# 30. Laplacian of Gaussian (LoG)
"""
The Laplacian of Gaussian (LoG) combines Gaussian smoothing and Laplacian edge detection.
"""
log_image = cv2.Laplacian(cv2.GaussianBlur(gray_image, (3, 3), 0), cv2.CV_64F)
