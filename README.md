# ML_Project_5793758

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Scripts Explanation](#scripts-explanation)
  - [convert_pdfs_to_images.py](#convert_pdfstoimagespy)
  - [ocr_images.py](#ocr_imagespy)
  - [clean_ocr_text.py](#clean_ocr_textpy)
  - [extract_geometric_features.py](#extract_geometric_featurespy)
  - [combine_ocr_and_geometric_features.py](#combine_ocr_and_geometric_featurespy)
  - [feature_engineering.py](#feature_engineeringpy)
  - [data_splitting.py](#data_splittingpy)
  - [custom_model_selection.py](#custom_model_selectionpy)
  - [train_evaluate_save_predictions.py](#train_evaluate_save_predictionspy)
- [Output Files](#output-files)
- [Contact](#contact)

## Description

My machine learning project aims to extract text and geometric features from construction plans using Optical Character Recognition (OCR) and simple image processing techniques. The extracted features are then used to classify the plans into predefined categories using machine learning models.

## Installation

1. **Clone the repository**:
   ```sh
   git clone <repository-url>
   cd ML_Project_5793758
   ```

2. **Create a virtual environment**:
   ```sh
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Data Preprocessing

**a. Convert PDFs to Images**:
```sh
python src/convert_pdfs_to_images.py
```

**b. Perform OCR on Images**:
```sh
python src/ocr_images.py
```

**c. Clean OCR Text**:
```sh
python src/clean_ocr_text.py
```

**d. Extract Geometric Features**:
```sh
python src/extract_geometric_features.py
```

**e. Combine OCR and Geometric Features**:
```sh
python src/combine_ocr_and_geometric_features.py
```

**f. Feature Engineering**:
```sh
python src/feature_engineering.py
```

### Step 2: Data Splitting

**Split the Dataset**:
```sh
python src/data_splitting.py
```

### Step 3: Model Selection

**Evaluate Multiple Classifiers**:
```sh
python src/custom_model_selection.py
```

### Step 4: Train and Evaluate the Best-Performing Model

**Train and Evaluate the Best Model**:
```sh
python src/train_evaluate_save_predictions.py
```

## Project Structure

```
ML_Project_5793758/
│
├── .idea/
│
├── .venv/
│
├── data/
│   ├── plans_pdfs/
│   ├── plans_imgs/
│   ├── ocr_text/
│   ├── cleaned_ocr_text/
│   ├── geometric_features/
│   ├── train_test_split/
│   ├── combined_dataset.csv
│   ├── feature_engineered_dataset.csv
│   ├── testing_data_with_predictions.csv
│
├── models/
│   ├── best_model_gradient_boosting.joblib
│
├── src/
│   ├── convert_pdfs_to_images.py
│   ├── ocr_images.py
│   ├── clean_ocr_text.py
│   ├── extract_geometric_features.py
│   ├── combine_ocr_and_geometric_features.py
│   ├── feature_engineering.py
│   ├── data_splitting.py
│   ├── custom_model_selection.py
│   ├── train_evaluate_save_predictions.py
│
└── README.md
```

## Dependencies

- Python 3.6+
- pdf2image
- pytesseract
- Pillow
- OpenCV
- numpy
- pandas
- scikit-learn
- joblib

## Scripts Explanation

### convert_pdfs_to_images.py

Converts PDF files in `data/plans_pdfs/` to processed images in `data/plans_imgs/`.

### ocr_images.py

Performs OCR on images in `data/plans_imgs/` and saves the text in `data/ocr_text/`.

### clean_ocr_text.py

Cleans the OCR text files in `data/ocr_text/` and saves the cleaned text in `data/cleaned_ocr_text/`.

### extract_geometric_features.py

Extracts geometric features from images in `data/plans_imgs/` and saves the features in `data/geometric_features/`.

### combine_ocr_and_geometric_features.py

Combines the cleaned OCR text and geometric features into a single CSV file `data/combined_dataset.csv`.

### feature_engineering.py

Performs feature engineering on the combined dataset and saves the resulting dataset in `data/feature_engineered_dataset.csv`.

### data_splitting.py

Splits the feature engineered dataset into training and testing sets and saves them in `data/train_test_split/`.

### custom_model_selection.py

Evaluates multiple classifiers on the training data and saves the results in `data/custom_model_results.csv`.

### train_evaluate_save_predictions.py

Trains the best-performing model, evaluates its performance, and saves the testing data with actual and predicted labels in `data/testing_data_with_predictions.csv`.

## Output Files

- **Combined Dataset**: `data/combined_dataset.csv`
- **Feature Engineered Dataset**: `data/feature_engineered_dataset.csv`
- **Model Results**: `data/custom_model_results.csv`
- **Testing Data with Predictions**: `data/testing_data_with_predictions.csv`
- **Saved Model**: `models/best_model_gradient_boosting.joblib`

## Contact

For any questions or issues, please contact Benjamin Ramos at [bramo029@fiu.edu](mailto:bramo029@fiu.edu).

---

Feel free to customize the repository URL and any other details as needed. This README file should provide a comprehensive guide to your project. Let me know if you need any further assistance!