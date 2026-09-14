# Diabetes Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts the likelihood of diabetes based on patient health parameters. The project includes data preprocessing, exploratory analysis, comparison of multiple classification models, and an interactive Streamlit dashboard for real-time predictions.

> **Disclaimer:** This project is created for educational and portfolio purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Features

- Predicts the likelihood of diabetes based on 8 patient health parameters
- Handles invalid zero values using median imputation
- Compares multiple machine learning classification models
- Uses Gaussian Naive Bayes as the final selected model
- Displays prediction probabilities for both outcomes
- Interactive Streamlit dashboard with real-time input controls
- Visualizes patient input using a clinical profile radar chart
- Includes informational health parameter summaries and guidance
- Provides model insights, evaluation metrics, and dataset information

## Dataset

This project uses the **Pima Indians Diabetes Dataset**, which contains medical diagnostic measurements used to predict whether a patient is likely to have diabetes.

### Dataset Details

- Total samples: 768
- Input features: 8
- Target variable: `Outcome`
- Classification type: Binary Classification

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | Diabetes outcome (0 or 1) |

## Machine Learning Workflow

### 1. Data Preprocessing

The dataset was cleaned before model training.

- Invalid zero values were identified in features where zero is not medically meaningful
- Zero values were replaced with missing values (`NaN`)
- Missing values were handled using median imputation

The affected features were:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

### 2. Train-Test Split

The dataset was divided into training and testing sets using an 80/20 split with stratification to preserve the class distribution.

### 3. Model Comparison

Multiple classification algorithms were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Models were compared using:

- Training Accuracy
- Test Accuracy
- Precision
- Recall
- F1 Score

### 4. Final Model

Gaussian Naive Bayes was selected as the final model based on its overall test performance and generalization.

The final model was saved as:

`diabetes_model.pkl`

## Model Performance

The final Gaussian Naive Bayes model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Test Accuracy | 72.7% |
| Precision | 74% |
| Recall | 80% |
| F1 Score | 77% |

### Confusion Matrix

```text
[[42 24]
 [18 70]]
 ```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook

## Project Structure

```text
Diabetes-Prediction/
│
├── app.py                  # Streamlit web application
├── diabetes.ipynb          # Data analysis and model development
├── diabetes.csv            # Dataset
├── diabetes_model.pkl      # Trained Gaussian Naive Bayes model
├── requirements.txt        # Project dependencies
├── .gitignore              # Files ignored by Git
└── README.md               # Project documentation
 ```
## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/KhushPawar/diabetes-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd diabetes-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser, where you can enter patient health parameters and view the model prediction.
## 🚀 Live Demo

[Try the Diabetes Prediction Dashboard](https://diabetes-prediction-7k7hezntgsmfbu7cvhv6rl.streamlit.app)
