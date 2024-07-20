from pdf2image import convert_from_path
import os
import time

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory where PDFs are stored
pdf_dir = os.path.join(project_root, 'data/plans_pdfs')
# Directory to save raw images
image_dir = os.path.join(project_root, 'data/raw_imgs')

# Create directories if they don't exist
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

print(f"PDF directory: {pdf_dir}")
print(f"Image directory: {image_dir}")

start_time = time.time()

# Convert each PDF to raw images
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
total_pdfs = len(pdf_files)
print(f"Total PDF files: {total_pdfs}")

for idx, pdf_file in enumerate(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    # Convert PDF to images
    print(f"Processing {pdf_file} ({idx + 1}/{total_pdfs})...")
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        # Save the raw image
        image_path = os.path.join(image_dir, f"{os.path.splitext(pdf_file)[0]}_page_{i + 1}.png")
        image.save(image_path, 'PNG')
    print(f"Converted and saved images for {pdf_file} ({idx + 1}/{total_pdfs})")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"PDFs have been successfully converted to images in {elapsed_time:.2f} seconds.")