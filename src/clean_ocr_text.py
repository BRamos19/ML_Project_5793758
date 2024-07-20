import os
import re
import chardet

# Directory where OCR results are stored
ocr_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ocr_results')

# Directory to save cleaned OCR results
cleaned_ocr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cleaned_ocr_results')
os.makedirs(cleaned_ocr_dir, exist_ok=True)

# Function to clean OCR text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-z0-9.,:;!?\'"()\[\]{}\-\/\\ ]', '', text)  # Remove special characters
    return text

# Function to detect encoding and read file
def read_file_with_encoding(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] if result['encoding'] is not None else 'utf-8'
    return raw_data.decode(encoding, errors='ignore')

# Clean each OCR result file
ocr_files = [f for f in os.listdir(ocr_results_dir) if f.endswith('.txt')]
total_files = len(ocr_files)
print(f"Total OCR files: {total_files}")

for idx, ocr_file in enumerate(ocr_files):
    ocr_file_path = os.path.join(ocr_results_dir, ocr_file)
    text = read_file_with_encoding(ocr_file_path)
    cleaned_text = clean_text(text)
    cleaned_ocr_path = os.path.join(cleaned_ocr_dir, ocr_file)
    with open(cleaned_ocr_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    print(f"Cleaned OCR text for {ocr_file} ({idx + 1}/{total_files})")

print(f"Total cleaned OCR files: {total_files}")