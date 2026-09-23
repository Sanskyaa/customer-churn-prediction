# 📊 Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer is likely to churn based on their demographic, service, contract, and billing information.

The project covers the complete machine learning workflow — from data understanding and preprocessing to model training, hyperparameter tuning, model explainability, and deployment using Streamlit.

---

## 🚀 Live Demo

🔗 **Live App:**  
https://YOUR-APP-URL.streamlit.app

> Replace the URL above with your actual Streamlit deployment URL after deployment.

---

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses.

The objective of this project is to build a machine learning model that predicts the probability of a customer leaving the service.

The project uses the **Telco Customer Churn dataset** and follows an end-to-end machine learning pipeline:

Raw Data
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Threshold Optimization
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
Deployment


🎯 Project Objectives
Understand the factors associated with customer churn
Clean and preprocess customer data
Perform exploratory data analysis
Engineer meaningful features
Train multiple classification models
Compare model performance
Tune the best-performing models
Optimize the classification threshold
Explain model predictions using SHAP
Build an interactive prediction application
Deploy the application for public access
📂 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information related to:

Demographics
Account information
Services
Contract details
Payment methods
Monthly charges
Total charges
Customer churn
Target Variable
Churn

Where:

Yes → Customer churned
No  → Customer did not churn
🛠️ Technologies Used
Programming Language
Python
Data Analysis
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
Model Explainability
SHAP
Visualization
Matplotlib
Seaborn
Web Application
Streamlit
Model Persistence
Joblib
Development Environment
Anaconda
Jupyter Notebook
Deployment
Streamlit Community Cloud
🤖 Machine Learning Models

Several classification algorithms were evaluated:

Logistic Regression
Decision Tree
Random Forest
XGBoost

The models were compared using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
📈 Model Performance
Baseline Model Comparison
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
XGBoost	79.15%	63.86%	48.92%	55.40%	0.841
Logistic Regression	74.16%	50.79%	77.69%	61.42%	0.840
Random Forest	76.87%	55.83%	60.48%	58.06%	0.824
Decision Tree	73.17%	49.32%	48.92%	49.12%	0.654
Tuned Models
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Tuned Logistic Regression	74.16%	50.79%	77.96%	61.51%	0.840
Tuned XGBoost	80.71%	67.84%	51.61%	58.63%	0.844

Metrics were calculated on the held-out test set. Results may vary if the data or preprocessing pipeline is changed.

⚙️ Hyperparameter Tuning

RandomizedSearchCV with 5-fold cross-validation was used to tune the XGBoost model.

The selected parameters were:

{
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
🎚️ Threshold Optimization

The default classification threshold of 0.50 was evaluated along with alternative thresholds.

A threshold of:

0.30

was selected during threshold analysis because it produced a higher F1 score in the evaluation performed for this project.

At the 0.30 threshold:

Accuracy  : 76.01%
Precision : 53.35%
Recall    : 75.00%
F1 Score  : 62.35%

The threshold can be adjusted depending on the business objective.

For example, a business may prefer higher recall if identifying more potentially churning customers is more important than minimizing false positives.

🧠 Feature Engineering

Several additional features were created to provide the model with more useful information.

Average Monthly Spend
AvgMonthlySpend = TotalCharges / Tenure
Tenure Group

Customers were grouped based on their tenure:

New
Growing
Mature
Loyal
Number of Services

Counts the number of subscribed services.

High-Value Customer

Identifies customers whose monthly charges are above the selected threshold.

Protection Indicator

Indicates whether the customer has at least one protection-related service.

🔍 Model Explainability

SHAP was used to understand how different features influence model predictions.

The SHAP analysis provides:

Global feature importance
Direction of feature impact
Individual customer explanations

This helps answer:

"Why does the model think this customer is likely to churn?"

🌐 Streamlit Application

The project includes an interactive Streamlit application.

Users can enter customer information such as:

Gender
Senior citizen status
Partner
Dependents
Tenure
Phone service
Internet service
Online security
Online backup
Device protection
Technical support
Streaming services
Contract type
Payment method
Monthly charges
Total charges

The application then returns:

Churn probability
Churn / No Churn prediction
Customer risk indication
📁 Project Structure
customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── telco_cleaned.csv
│
├── models/
│   ├── churn_model.joblib
│   └── churn_threshold.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_tuning.ipynb
│   └── 05_model_evaluation.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
💻 Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
2. Navigate to the project
cd customer-churn-prediction
3. Create the Conda environment
conda create -n churn_env python=3.11
4. Activate the environment
conda activate churn_env
5. Install dependencies
pip install -r requirements.txt
▶️ Run the Application Locally

From the project root:

streamlit run app/app.py

The application will open in your browser.

📊 Notebooks

The notebooks are organized according to the machine learning workflow.

01_data_understanding.ipynb
Dataset loading
Dataset structure
Data types
Missing values
Duplicate detection
Target distribution
Initial data exploration
02_preprocessing.ipynb
Data cleaning
Data type conversion
Feature engineering
Encoding
Preprocessing pipeline
03_model_training.ipynb
Train/test split
Model training
Baseline model comparison
Evaluation metrics
04_model_tuning.ipynb
Cross-validation
Hyperparameter tuning
RandomizedSearchCV
Best parameter selection
05_model_evaluation.ipynb
Final model evaluation
Threshold optimization
Confusion matrix
ROC curve
Precision-Recall curve
SHAP explainability
Model saving
📦 Requirements

Main dependencies:

streamlit
pandas
numpy
scikit-learn
xgboost
joblib
🔮 Future Improvements

Possible improvements for future versions include:

Cost-sensitive threshold optimization
Customer retention recommendations
Interactive SHAP explanations in the web app
Model monitoring
Automated retraining
Additional models such as LightGBM
Customer segmentation
Churn risk dashboard
Integration with a database
CI/CD pipeline
⚠️ Limitations
The model is trained on a public Telco customer dataset.
Model performance depends on the quality and representativeness of the dataset.
Predictions should be treated as decision-support outputs rather than guaranteed outcomes.
The selected classification threshold is based on the evaluation performed in this project.
🎓 Project Type

College Mini Project

Domain: Machine Learning / Data Science

Task: Binary Classification

Application: Customer Churn Prediction

👨‍💻 Author

Sanskar Waskar

Final Year Computer Engineering Student

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/
