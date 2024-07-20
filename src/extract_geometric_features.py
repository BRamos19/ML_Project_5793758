import cv2
import os
import numpy as np
import time

# Directory where processed images are stored
image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/plans_imgs')
# Directory to save geometric feature results
geometric_features_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'data/geometric_features')
os.makedirs(geometric_features_dir, exist_ok=True)

print(f"Processed image directory: {image_dir}")
print(f"Geometric features directory: {geometric_features_dir}")


def extract_geometric_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    num_lines = len(lines) if lines is not None else 0

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    # Shape detection (rectangles)
    num_rectangles = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # If the contour has 4 vertices, it is likely a rectangle
            num_rectangles += 1

    # Corner detection using Harris Corner Detector
    corners = cv2.cornerHarris(image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    threshold = 0.01 * corners.max()
    num_corners = np.sum(corners > threshold)

    # Detect text areas using contour properties
    num_text_areas = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 5.0:  # Typical aspect ratio range for text
            num_text_areas += 1

    return num_lines, num_contours, num_rectangles, num_corners, num_text_areas


start_time = time.time()

# Extract geometric features from each processed image
image_files = [f for f in os.listdir(image_dir) if f.startswith('processed_') and f.endswith('.png')]
total_images = len(image_files)
print(f"Total processed images: {total_images}")

geometric_features = []

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    num_lines, num_contours, num_rectangles, num_corners, num_text_areas = extract_geometric_features(image_path)
    geometric_features.append((image_file, num_lines, num_contours, num_rectangles, num_corners, num_text_areas))
    print(f"Geometric features extracted for {image_file} ({idx + 1}/{total_images})")

# Save the geometric features to a file
geometric_features_path = os.path.join(geometric_features_dir, 'geometric_features.txt')
with open(geometric_features_path, 'w') as f:
    for image_file, num_lines, num_contours, num_rectangles, num_corners, num_text_areas in geometric_features:
        f.write(f"{image_file}\t{num_lines}\t{num_contours}\t{num_rectangles}\t{num_corners}\t{num_text_areas}\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Geometric feature extraction completed for all images in {elapsed_time:.2f} seconds.")