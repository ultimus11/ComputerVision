import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
image = cv2.imread('image1.jpg')

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Display the images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(122),plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)),plt.title('Gaussian Blurred')
plt.show()

####################################################################################################################################
####################################################################################################################################

# Apply Median Blur
median_blurred_image = cv2.medianBlur(image, 5)

# Display the images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(122),plt.imshow(cv2.cvtColor(median_blurred_image, cv2.COLOR_BGR2RGB)),plt.title('Median Blurred')
plt.show()

####################################################################################################################################
####################################################################################################################################

# Apply Bilateral Filter
bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)

# Display the images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(122),plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB)),plt.title('Bilateral Filtered')
plt.show()

####################################################################################################################################
####################################################################################################################################

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Sobel Filter
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

# Display the images
plt.subplot(131),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(132),plt.imshow(sobelx, cmap='gray'),plt.title('Sobel X')
plt.subplot(133),plt.imshow(sobely, cmap='gray'),plt.title('Sobel Y')
plt.show()

####################################################################################################################################
####################################################################################################################################

# Apply Laplacian Filter
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# Display the images
plt.subplot(121),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(122),plt.imshow(laplacian, cmap='gray'),plt.title('Laplacian')
plt.show()


####################################################################################################################################
####################################################################################################################################

# Apply Canny Edge Detector
edges = cv2.Canny(image, 100, 200)

# Display the images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(122),plt.imshow(edges, cmap='gray'),plt.title('Canny Edges')
plt.show()


####################################################################################################################################
####################################################################################################################################


# Apply Scharr Filter
scharrx = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)

# Display the images
plt.subplot(131),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(132),plt.imshow(scharrx, cmap='gray'),plt.title('Scharr X')
plt.subplot(133),plt.imshow(scharry, cmap='gray'),plt.title('Scharr Y')
plt.show()


####################################################################################################################################
####################################################################################################################################

from skimage.filters import gabor

# Apply Gabor Filter
frequency = 0.6
filtered, _ = gabor(gray_image, frequency=frequency)

# Display the images
plt.subplot(121),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(122),plt.imshow(filtered, cmap='gray'),plt.title('Gabor Filter')
plt.show()

####################################################################################################################################
####################################################################################################################################

# Apply Non-Local Means Denoising
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Display the images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(122),plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)),plt.title('Denoised')
plt.show()


####################################################################################################################################
####################################################################################################################################

from skimage import filters

# Apply Unsharp Masking
unsharp_image = filters.unsharp_mask(gray_image, radius=1.0, amount=1.0)

# Display the images
plt.subplot(121),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(122),plt.imshow(unsharp_image, cmap='gray'),plt.title('Unsharp Masking')
plt.show()


####################################################################################################################################
####################################################################################################################################

# Convert to grayscale if needed
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray_image)

# Display the images
plt.subplot(121),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(122),plt.imshow(clahe_image, cmap='gray'),plt.title('CLAHE')
plt.show()


####################################################################################################################################
####################################################################################################################################

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(gray_image)

# Display the images
plt.subplot(121),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(122),plt.imshow(equalized_image, cmap='gray'),plt.title('Equalized')
plt.show()


####################################################################################################################################
####################################################################################################################################

# Apply Morphological Operations
kernel = np.ones((5,5), np.uint8)

# Erosion
erosion = cv2.erode(gray_image, kernel, iterations = 1)

# Dilation
dilation = cv2.dilate(gray_image, kernel, iterations = 1)

# Opening
opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

# Closing
closing = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

# Display the images
plt.subplot(141),plt.imshow(erosion, cmap='gray'),plt.title('Erosion')
plt.subplot(142),plt.imshow(dilation, cmap='gray'),plt.title('Dilation')
plt.subplot(143),plt.imshow(opening, cmap='gray'),plt.title('Opening')
plt.subplot(144),plt.imshow(closing, cmap='gray'),plt.title('Closing')
plt.show()


####################################################################################################################################
####################################################################################################################################

# Apply Fourier Transform
dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Magnitude Spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Display the images
plt.subplot(121),plt.imshow(gray_image, cmap='gray'),plt.title('Original')
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap='gray'),plt.title('Magnitude Spectrum')
plt.show()


####################################################################################################################################
####################################################################################################################################

import cv2.ximgproc as xip

# Apply Guided Filter
guided_filter = xip.guidedFilter(image, image, radius=8, eps=0.1)

# Display the images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(122),plt.imshow(cv2.cvtColor(guided_filter, cv2.COLOR_BGR2RGB)),plt.title('Guided Filter')
plt.show()


####################################################################################################################################
####################################################################################################################################
