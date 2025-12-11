# Customer Churn Prediction Platform

## Overview
A comprehensive customer churn prediction platform built with Streamlit that helps businesses identify at-risk customers and take proactive retention actions. The platform features multiple ML models, interactive visualizations, customer segmentation, CLV prediction, A/B testing framework, and actionable retention recommendations.

## Current State
- **Status**: Fully functional with advanced features
- **Last Updated**: December 2024
- **Tech Stack**: Python 3.11, Streamlit, Scikit-learn, XGBoost, Plotly, PostgreSQL, SQLAlchemy

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
- Simple oversampling for class imbalance handling

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
- Predictions saved to database for history tracking

### 6. Customer Segmentation
- K-means clustering for customer grouping
- Segment visualization and profiling
- Feature heatmaps by segment
- Downloadable segmented data

### 7. Prediction History (NEW)
- Database-backed prediction storage
- View recent predictions with risk distribution
- Track prediction trends over time
- Risk level analytics

### 8. Customer Lifetime Value (CLV) Prediction (NEW)
- CLV estimation based on churn probability
- Configurable revenue and margin parameters
- CLV segmentation (Very Low to Very High)
- CLV vs Churn probability visualization
- Downloadable CLV predictions

### 9. A/B Testing Framework (NEW)
- Create retention strategy experiments
- Define control and treatment groups
- Set target segments and sample sizes
- Track experiment results
- Simulated outcome visualization

## Project Structure
```
/
├── app.py                 # Main Streamlit application
├── models.py              # Database models (SQLAlchemy)
├── data/
│   └── sample_churn_data.csv  # Sample dataset for testing
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── pyproject.toml         # Python dependencies
└── READ.md              # This file
```

## Database Schema
- **prediction_history**: Stores individual customer predictions
- **model_versions**: Tracks trained model versions and performance
- **ab_experiments**: A/B testing experiment configurations
- **ab_experiment_results**: Results for A/B experiments
- **data_drift_logs**: Data drift detection results
- **clv_predictions**: Customer lifetime value predictions

## Running the Application
The app runs on port 5000 using:
streamlit run app.py 

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- plotly
- matplotlib
- seaborn
- psycopg2-binary
- sqlalchemy

## Environment Variables
- DATABASE_URL: PostgreSQL connection string
- PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE: Individual PostgreSQL credentials

## User Preferences
- Clean, professional UI with tabbed navigation
- Interactive Plotly visualizations preferred over static matplotlib
- Business-friendly language in recommendations
- Downloadable reports for all major outputs
- Database persistence for predictions and experiments

## Future Enhancements
- Real-time prediction API endpoint for production integration
- Automated retraining pipeline with data drift detection
- Email notifications for high-risk predictions
- Integration with CRM systems
- Advanced CLV models using BG/NBD framework
