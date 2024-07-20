import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import json
import joblib

# Load the split datasets
train_test_split_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                                    'train_test_split')
X_train = pd.read_csv(os.path.join(train_test_split_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(train_test_split_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(train_test_split_dir, 'y_train.csv')).squeeze()
y_test = pd.read_csv(os.path.join(train_test_split_dir, 'y_test.csv')).squeeze()

# Define classifiers to evaluate
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

# Evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Classification Report": report
    })
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                            'custom_model_results.csv')
results_df.to_csv(results_path, index=False)

# Identify the best model
best_model_info = max(results, key=lambda x: x['Accuracy'])
print(f"Best Model: {best_model_info['Model']}")
print(f"Best Model Accuracy: {best_model_info['Accuracy']}")
print(f"Best Model Classification Report:\n{best_model_info['Classification Report']}")

# Save the best model information to a JSON file
best_model_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                                    'best_model_info.json')
with open(best_model_info_path, 'w') as f:
    json.dump(best_model_info, f)

# Save the best model
best_model = classifiers[best_model_info['Model']]
best_model.fit(X_train, y_train)
best_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models',
                               f"best_model_{best_model_info['Model'].replace(' ', '_').lower()}.joblib")
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
joblib.dump(best_model, best_model_path)
print(f"Best model saved to: {best_model_path}")

print("Model evaluation completed.")