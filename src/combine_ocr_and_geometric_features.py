import os
import re
import pandas as pd
from collections import Counter

# Paths
cleaned_ocr_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/cleaned_ocr_results')
geometric_features_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/geometric_features')
manual_labeled_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/manual_labeled_data.csv')

# Read cleaned OCR results
ocr_data = []
for ocr_file in os.listdir(cleaned_ocr_results_dir):
    if ocr_file.endswith('.txt'):
        with open(os.path.join(cleaned_ocr_results_dir, ocr_file), 'r') as f:
            text = f.read()
        ocr_data.append({'ImageFile': ocr_file.replace('.txt', '.png'), 'OCRText': text})

ocr_df = pd.DataFrame(ocr_data)

# Read geometric features
geometric_features = []
with open(os.path.join(geometric_features_dir, 'geometric_features.txt'), 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        image_file = parts[0]
        num_lines, num_contours, num_rectangles, num_corners, num_text_areas = map(int, parts[1:])
        geometric_features.append({
            'ImageFile': image_file,
            'NumLines': num_lines,
            'NumContours': num_contours,
            'NumRectangles': num_rectangles,
            'NumCorners': num_corners,
            'NumTextAreas': num_text_areas
        })

geometric_features_df = pd.DataFrame(geometric_features)

# Merge OCR results with geometric features
combined_df = pd.merge(ocr_df, geometric_features_df, on='ImageFile')

# Define the keywords and their corresponding labels
keywords = {
    'Construction Sheet': ['architectural', 'arch', 'a', 'fl', 'rm', 'wdw', 'dr', 'clg',
                           'structural', 'struc', 's', 'bm', 'col', 'ftg', 'slb', 'reinf',
                           'civil', 'c', 'grd', 'drn', 'rd', 'swk', 'clvt'],
    'MEP Sheet': ['mechanical', 'mech', 'm', 'hvac', 'dct', 'fan', 'uh', 'chlr',
                  'plumbing', 'p', 'plumb', 'pipe', 'val', 'wc', 'sk', 'wh',
                  'electrical', 'elec', 'e', 'ckt', 'ltg', 'olt', 'sw', 'v'],
    'Code/Spec Sheet': ['code', 'specifications', 'specs', 'spec', 'bc', 'fc', 'ec', 'zc']
}

# Function to add predicted labels based on the most frequent keyword
def add_predicted_labels(row):
    text = row['OCRText']
    word_counts = Counter()
    for label, words in keywords.items():
        for word in words:
            word_counts[label] += text.split().count(word)
    if not word_counts:
        return ''  # Leave empty if no keywords match
    most_common_label, count = word_counts.most_common(1)[0]
    return most_common_label if count > 0 else ''

# Add the 'p.label' column with predicted labels
combined_df['P.Label'] = combined_df.apply(add_predicted_labels, axis=1)

# Read the manually labeled data
manual_labeled_df = pd.read_csv(manual_labeled_data_path)

# Merge the manual labels into the combined dataset
combined_df = pd.merge(combined_df, manual_labeled_df, on='ImageFile', how='left')

# Extract the numerical parts for sorting
def extract_sort_keys(image_file):
    matches = re.findall(r'\d+', image_file)
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    elif len(matches) == 1:
        return int(matches[0]), 0
    else:
        return 0, 0

combined_df['SortKey'] = combined_df['ImageFile'].apply(lambda x: extract_sort_keys(x))

# Sort the combined dataset by SortKey and ImageFile
combined_df_sorted = combined_df.sort_values(by=['SortKey', 'ImageFile'])

# Remove the SortKey column after sorting
combined_df_sorted = combined_df_sorted.drop(columns=['SortKey'])

# Display the combined DataFrame
print(combined_df_sorted)

# Save the combined dataset to a CSV file for later use
combined_df_sorted.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'combined_dataset.csv'), index=False)

# Print count of each label
label_counts = combined_df_sorted['Label'].value_counts()
print("\nLabel counts:")
print(label_counts)