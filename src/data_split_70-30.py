import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the feature engineered dataset
feature_engineered_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'feature_engineered_dataset.csv')

# Load the feature engineered dataset
df = pd.read_csv(feature_engineered_dataset_path)

# Define the features and the target
X = df.drop(columns=['Label', 'p.label', 'ImageFile', 'OCRText'])  # Drop non-feature columns
y = df['Label']  # Use the manual labels as the target

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the split datasets
train_test_split_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train_test_split')
os.makedirs(train_test_split_dir, exist_ok=True)

X_train.to_csv(os.path.join(train_test_split_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(train_test_split_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(train_test_split_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(train_test_split_dir, 'y_test.csv'), index=False)

print("Data splitting completed.")