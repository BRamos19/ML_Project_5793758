import pytesseract
import os
import time

# Directory where processed images are stored
image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/plans_imgs')

# Directory to save OCR results
ocr_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/ocr_results')
os.makedirs(ocr_results_dir, exist_ok=True)

start_time = time.time()

# Extract text from each processed image
image_files = [f for f in os.listdir(image_dir) if f.startswith('processed_') and f.endswith('.png')]
total_images = len(image_files)
print(f"Total processed images: {total_images}")

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)

    # Use Tesseract to do OCR on the processed image
    text = pytesseract.image_to_string(image_path)

    # Save the OCR result
    ocr_result_path = os.path.join(ocr_results_dir, f"{os.path.splitext(image_file)[0]}.txt")
    with open(ocr_result_path, 'w') as f:
        f.write(text)
    print(f"OCR completed for {image_file} ({idx + 1}/{total_images})")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"OCR process completed for all images in {elapsed_time:.2f} seconds.")