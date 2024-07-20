import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Path to the combined dataset
combined_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'combined_dataset.csv')

# Load the combined dataset
combined_df = pd.read_csv(combined_dataset_path)

# Define keywords for feature extraction
keywords = {
    'Construction Sheet': ['architectural', 'arch', 'a', 'fl', 'rm', 'wdw', 'dr', 'clg',
                           'structural', 'struc', 's', 'bm', 'col', 'ftg', 'slb', 'reinf',
                           'civil', 'c', 'grd', 'drn', 'rd', 'swk', 'clvt'],
    'MEP Sheet': ['mechanical', 'mech', 'm', 'hvac', 'dct', 'fan', 'uh', 'chlr',
                  'plumbing', 'p', 'plumb', 'pipe', 'val', 'wc', 'sk', 'wh',
                  'electrical', 'elec', 'e', 'ckt', 'ltg', 'olt', 'sw', 'v'],
    'Code/Spec Sheet': ['code', 'specifications', 'specs', 'spec', 'bc', 'fc', 'ec', 'zc']
}

# Create keyword count features
for category, words in keywords.items():
    for word in words:
        combined_df[f'count_{word}'] = combined_df['OCRText'].apply(lambda x: x.split().count(word))

# Normalize geometric features
geometric_features = ['NumLines', 'NumContours', 'NumRectangles', 'NumCorners', 'NumTextAreas']
scaler = StandardScaler()
combined_df[geometric_features] = scaler.fit_transform(combined_df[geometric_features])

# Create text length feature
combined_df['text_length'] = combined_df['OCRText'].apply(len)

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['OCRText'])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate TF-IDF features with the combined dataframe
combined_df = pd.concat([combined_df, tfidf_df], axis=1)

# Save the dataset with engineered features
feature_engineered_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'feature_engineered_dataset.csv')
combined_df.to_csv(feature_engineered_dataset_path, index=False)

# Display the first few rows of the new dataset
print(combined_df.head())