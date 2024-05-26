import cv2
import numpy as np

def canny_edge_detection(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply GaussianBlur to reduce image noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # Display the result
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough_line_transform(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    colored_image = cv2.imread(image_path)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Draw lines on the image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(colored_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Hough Line Transform', colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce image noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Contour Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'outgen1716638200.jpg'
    
    print("Canny Edge Detection")
    canny_edge_detection(image_path)
    
    print("\nHough Line Transform")
    hough_line_transform(image_path)
    
    print("\nContour Detection")
    contour_detection(image_path)
    
