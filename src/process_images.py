import cv2
import os
import time

# Directory where raw images are stored
raw_image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/raw_imgs')
# Directory to save processed images
processed_image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/plans_imgs')

# Create directories if they don't exist
os.makedirs(raw_image_dir, exist_ok=True)
os.makedirs(processed_image_dir, exist_ok=True)

print(f"Raw image directory: {raw_image_dir}")
print(f"Processed image directory: {processed_image_dir}")

# Preprocessing function
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    equalized_image = cv2.equalizeHist(gray_image)
    # Reduce noise
    denoised_image = cv2.fastNlMeansDenoising(equalized_image, h=30)
    return denoised_image

start_time = time.time()

# Process each raw image
image_files = [f for f in os.listdir(raw_image_dir) if f.endswith('.png')]
total_images = len(image_files)
print(f"Total raw images: {total_images}")

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(raw_image_dir, image_file)
    # Load the raw image
    raw_image = cv2.imread(image_path)
    # Preprocess the image
    processed_image = preprocess_image(raw_image)
    # Save the processed image
    processed_image_path = os.path.join(processed_image_dir, f"processed_{image_file}")
    cv2.imwrite(processed_image_path, processed_image)
    print(f"Processed and saved image for {image_file} ({idx + 1}/{total_images})")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Image processing completed for all images in {elapsed_time:.2f} seconds.")