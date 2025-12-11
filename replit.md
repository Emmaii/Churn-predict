# Customer Churn Prediction Platform

## Overview
An enhanced customer churn prediction platform built with Streamlit that helps businesses identify at-risk customers and take proactive retention actions. The platform features multiple ML models, interactive visualizations, customer segmentation, and actionable retention recommendations.

## Current State
- **Status**: Fully functional
- **Last Updated**: December 2024
- **Tech Stack**: Python 3.11, Streamlit, Scikit-learn, XGBoost, Plotly

## Features

### 1. Data Upload & Profiling
- CSV file upload with automatic validation
- Comprehensive data profiling (missing values, distributions, correlations)
- Column information and data type detection
- Interactive distribution visualizations

### 2. Data Preprocessing
- Automatic handling of missing values
- Duplicate removal
- Label encoding for categorical variables
- Feature engineering (sum, mean, std of numeric features)
- Standard scaling for model training
- SMOTE for class imbalance handling

### 3. Model Training
- Multiple ML algorithms:
  - Random Forest
  - XGBoost
  - Logistic Regression
- Cross-model performance comparison
- ROC curve visualization
- Feature importance analysis using permutation importance

### 4. Analytics Dashboard
- Key performance metrics (Accuracy, Precision, Recall, AUC-ROC)
- Confusion matrix visualization
- Churn probability distribution
- Risk segmentation (High/Medium/Low)
- Downloadable performance reports

### 5. Individual Predictions
- Customer-level churn probability prediction
- Visual risk gauge
- Actionable retention recommendations based on risk level

### 6. Customer Segmentation
- K-means clustering for customer grouping
- Segment visualization and profiling
- Feature heatmaps by segment
- Downloadable segmented data

## Project Structure
```
/
├── app.py                 # Main Streamlit application
├── data/
│   └── sample_churn_data.csv  # Sample dataset for testing
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── pyproject.toml         # Python dependencies
└── replit.md              # This file
```

## Running the Application
The app runs on port 5000 using:
```bash
streamlit run app.py --server.port 5000
```

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- plotly
- matplotlib
- seaborn
- imbalanced-learn

## User Preferences
- Clean, professional UI with tabbed navigation
- Interactive Plotly visualizations preferred over static matplotlib
- Business-friendly language in recommendations
- Downloadable reports for all major outputs

## Future Enhancements
- Database integration for historical predictions
- A/B testing framework for retention strategies
- Automated retraining pipeline
- Real-time prediction API
- Customer lifetime value (CLV) prediction
