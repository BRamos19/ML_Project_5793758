import os
import pandas as pd
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the split datasets
train_test_split_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train_test_split')
X_train = pd.read_csv(os.path.join(train_test_split_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(train_test_split_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(train_test_split_dir, 'y_train.csv')).squeeze()
y_test = pd.read_csv(os.path.join(train_test_split_dir, 'y_test.csv')).squeeze()

# Add the ImageFile column back to X_test for reference
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
combined_data_path = os.path.join(data_dir, 'combined_dataset.csv')
combined_df = pd.read_csv(combined_data_path)
X_test = X_test.join(combined_df[['ImageFile']])

# Load the best model information
best_model_info_path = os.path.join(data_dir, 'best_model_info.json')
with open(best_model_info_path, 'r') as f:
    best_model_info = json.load(f)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Select the best model
best_model_name = best_model_info['Model']
best_model = classifiers[best_model_name]

# Train the best model
print(f"Training the best model: {best_model_name}...")
best_model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
predictions = best_model.predict(X_test.drop(columns=['ImageFile']))
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the trained model
best_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', f"best_model_{best_model_name.replace(' ', '_').lower()}.joblib")
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
joblib.dump(best_model, best_model_path)
print(f"Best model saved to: {best_model_path}")

# Save testing data with predicted labels to CSV
output_df = pd.DataFrame({
    'ImageFile': X_test['ImageFile'],
    'Actual Label': y_test,
    'Predicted Label': predictions
})

output_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'testing_data_with_predictions.csv')
output_df.to_csv(output_csv_path, index=False)
print(f"Testing data with predictions saved to: {output_csv_path}")