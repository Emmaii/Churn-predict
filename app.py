import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def simple_oversample(X, y):
    """Simple oversampling to balance classes using numpy operations"""
    np.random.seed(42)
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    unique, counts = np.unique(y, return_counts=True)
    max_count = int(max(counts))
    
    X_parts = [X]
    y_parts = [y]
    
    for class_val in unique:
        class_indices = np.where(y == class_val)[0]
        class_count = len(class_indices)
        
        if class_count < max_count:
            n_samples_needed = max_count - class_count
            resampled_indices = np.random.choice(class_indices, size=n_samples_needed, replace=True)
            X_parts.append(X[resampled_indices])
            y_parts.append(y[resampled_indices])
    
    return np.vstack(X_parts), np.concatenate(y_parts)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data': None,
        'cleaned_data': None,
        'original_data': None,
        'models': {},
        'best_model': None,
        'best_model_name': None,
        'scaler': None,
        'label_encoders': {},
        'feature_columns': None,
        'target_column': None,
        'categorical_columns': [],
        'numeric_columns': [],
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'predictions': None,
        'model_metrics': {},
        'feature_importance': None,
        'cluster_labels': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_and_validate_data(uploaded_file):
    """Load CSV and perform initial validation"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


def get_data_profile(df):
    """Generate comprehensive data profile"""
    profile = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_total': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicates': df.duplicated().sum(),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    return profile


def preprocess_data(df, target_col):
    """Clean and preprocess the data"""
    df_clean = df.copy()
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    df_clean = df_clean.drop_duplicates()
    
    label_encoders = {}
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    if target_col in df_clean.columns:
        if df_clean[target_col].dtype == 'object':
            le = LabelEncoder()
            df_clean[target_col] = le.fit_transform(df_clean[target_col].astype(str))
            label_encoders[target_col] = le
    
    return df_clean, label_encoders, categorical_cols, numeric_cols


def engineer_features(df, target_col):
    """Add engineered features if applicable columns exist"""
    df_eng = df.copy()
    
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) >= 2:
        df_eng['feature_sum'] = df_eng[numeric_cols].sum(axis=1)
        df_eng['feature_mean'] = df_eng[numeric_cols].mean(axis=1)
        df_eng['feature_std'] = df_eng[numeric_cols].std(axis=1)
    
    return df_eng


def train_models(X_train, X_test, y_train, y_test, use_smote=True):
    """Train multiple models and return metrics"""
    
    if use_smote and len(np.unique(y_train)) == 2:
        try:
            X_train_balanced, y_train_balanced = simple_oversample(X_train, y_train)
        except:
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 0,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        trained_models[name] = model
    
    return trained_models, results


def get_feature_importance(model, X, y, feature_names):
    """Get feature importance using permutation importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        importance = result.importances_mean
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def perform_clustering(X, n_clusters=3):
    """Perform K-means clustering for customer segmentation"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans


def get_retention_recommendations(probability, feature_values, feature_names):
    """Generate retention recommendations based on churn probability and features"""
    recommendations = []
    
    if probability >= 0.7:
        recommendations.append("HIGH PRIORITY: Immediate intervention required")
        recommendations.append("Consider offering personalized discount or loyalty reward")
        recommendations.append("Schedule direct outreach from customer success team")
    elif probability >= 0.4:
        recommendations.append("MEDIUM PRIORITY: Proactive engagement recommended")
        recommendations.append("Send personalized retention offer")
        recommendations.append("Review recent customer interactions for issues")
    else:
        recommendations.append("LOW PRIORITY: Continue standard engagement")
        recommendations.append("Monitor for any changes in behavior")
        recommendations.append("Consider upsell opportunities")
    
    return recommendations


initialize_session_state()

st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

tabs = st.tabs(["📁 Data Upload", "🔧 Preprocessing", "🤖 Model Training", "📈 Analytics Dashboard", "🎯 Predictions", "👥 Segmentation"])

with tabs[0]:
    st.header("Data Upload & Profiling")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            df, error = load_and_validate_data(uploaded_file)
            
            if error:
                st.error(f"Error loading file: {error}")
            else:
                st.session_state.data = df
                st.success(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("Data Profile")
            profile = get_data_profile(st.session_state.data)
            
            st.metric("Total Rows", f"{profile['rows']:,}")
            st.metric("Total Columns", profile['columns'])
            st.metric("Missing Values", f"{profile['missing_pct']:.1f}%")
            st.metric("Duplicates", profile['duplicates'])
            st.metric("Memory Usage", f"{profile['memory_usage']:.2f} MB")
    
    if st.session_state.data is not None:
        st.subheader("Column Information")
        
        col_info = []
        for col in st.session_state.data.columns:
            col_info.append({
                'Column': col,
                'Type': str(st.session_state.data[col].dtype),
                'Non-Null Count': st.session_state.data[col].notna().sum(),
                'Null Count': st.session_state.data[col].isna().sum(),
                'Unique Values': st.session_state.data[col].nunique()
            })
        
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        st.subheader("Data Distribution")
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribution', 'Box Plot'))
            
            fig.add_trace(
                go.Histogram(x=st.session_state.data[selected_col], name='Distribution'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(y=st.session_state.data[selected_col], name='Box Plot'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.header("Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload tab.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            target_options = st.session_state.data.columns.tolist()
            churn_like_cols = [col for col in target_options if 'churn' in col.lower()]
            default_target = churn_like_cols[0] if churn_like_cols else target_options[0]
            
            target_col = st.selectbox(
                "Select Target Column (Churn indicator)",
                target_options,
                index=target_options.index(default_target) if default_target in target_options else 0
            )
            st.session_state.target_column = target_col
        
        with col2:
            use_feature_engineering = st.checkbox("Enable Feature Engineering", value=True)
            use_smote = st.checkbox("Use SMOTE for Class Balancing", value=True)
        
        if st.button("Preprocess Data", type="primary"):
            with st.spinner("Processing data..."):
                st.session_state.original_data = st.session_state.data.copy()
                
                df_clean, label_encoders, cat_cols, num_cols = preprocess_data(st.session_state.data, target_col)
                
                if use_feature_engineering:
                    df_clean = engineer_features(df_clean, target_col)
                
                st.session_state.cleaned_data = df_clean
                st.session_state.label_encoders = label_encoders
                st.session_state.categorical_columns = cat_cols
                st.session_state.numeric_columns = num_cols
                
                feature_cols = [col for col in df_clean.columns if col != target_col]
                st.session_state.feature_columns = feature_cols
                
                X = df_clean[feature_cols]
                y = df_clean[target_col]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.scaler = scaler
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("Data preprocessing completed!")
        
        if st.session_state.cleaned_data is not None:
            st.subheader("Preprocessing Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Rows", len(st.session_state.data))
            with col2:
                st.metric("Cleaned Rows", len(st.session_state.cleaned_data))
            with col3:
                st.metric("Features Created", len(st.session_state.feature_columns))
            
            st.subheader("Target Distribution")
            target_counts = st.session_state.cleaned_data[target_col].value_counts()
            
            fig = px.pie(
                values=target_counts.values,
                names=['Not Churned', 'Churned'] if len(target_counts) == 2 else target_counts.index.tolist(),
                title="Target Variable Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Correlation Matrix")
            numeric_data = st.session_state.cleaned_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlation Heatmap"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Download Cleaned Data")
            csv = st.session_state.cleaned_data.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv,
                file_name="cleaned_customer_data.csv",
                mime="text/csv"
            )

with tabs[2]:
    st.header("Model Training & Comparison")
    
    if st.session_state.X_train is None:
        st.warning("Please preprocess data first in the Preprocessing tab.")
    else:
        if st.button("Train Models", type="primary"):
            with st.spinner("Training multiple models... This may take a moment."):
                models, results = train_models(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_test,
                    use_smote=True
                )
                
                st.session_state.models = models
                st.session_state.model_metrics = results
                
                best_model_name = max(results, key=lambda x: results[x]['auc_roc'])
                st.session_state.best_model = models[best_model_name]
                st.session_state.best_model_name = best_model_name
                
                st.success(f"Models trained successfully! Best model: {best_model_name}")
        
        if st.session_state.model_metrics:
            st.subheader("Model Performance Comparison")
            
            metrics_df = pd.DataFrame(st.session_state.model_metrics).T
            metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']]
            
            st.dataframe(
                metrics_df.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            fig = go.Figure()
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
                fig.add_trace(go.Bar(
                    name=metric.upper().replace('_', '-'),
                    x=list(st.session_state.model_metrics.keys()),
                    y=[st.session_state.model_metrics[m][metric] for m in st.session_state.model_metrics]
                ))
            
            fig.update_layout(
                barmode='group',
                title="Model Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("ROC Curves")
            
            fig = go.Figure()
            
            for name, metrics in st.session_state.model_metrics.items():
                fpr, tpr, _ = roc_curve(st.session_state.y_test, metrics['probabilities'])
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={metrics['auc_roc']:.3f})",
                    mode='lines'
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature Importance")
            
            if st.session_state.best_model is not None:
                importance_df = get_feature_importance(
                    st.session_state.best_model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.feature_columns
                )
                
                fig = px.bar(
                    importance_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f"Top 15 Feature Importance ({st.session_state.best_model_name})",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.header("Analytics Dashboard")
    
    if not st.session_state.model_metrics:
        st.warning("Please train models first in the Model Training tab.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        best_metrics = st.session_state.model_metrics[st.session_state.best_model_name]
        
        with col1:
            st.metric("Accuracy", f"{best_metrics['accuracy']:.1%}")
        with col2:
            st.metric("Precision", f"{best_metrics['precision']:.1%}")
        with col3:
            st.metric("Recall", f"{best_metrics['recall']:.1%}")
        with col4:
            st.metric("AUC-ROC", f"{best_metrics['auc_roc']:.3f}")
        
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(st.session_state.y_test, best_metrics['predictions'])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Churned', 'Churned'],
            y=['Not Churned', 'Churned'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(title=f"Confusion Matrix - {st.session_state.best_model_name}", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Churn Probability Distribution")
        
        fig = go.Figure()
        
        probs = best_metrics['probabilities']
        actual = st.session_state.y_test
        
        fig.add_trace(go.Histogram(
            x=probs[actual == 0],
            name='Not Churned',
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=probs[actual == 1],
            name='Churned',
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.update_layout(
            barmode='overlay',
            title="Churn Probability Distribution by Actual Class",
            xaxis_title="Predicted Churn Probability",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Business Insights")
        
        total_customers = len(st.session_state.y_test)
        churned_customers = int(sum(best_metrics['predictions']))
        high_risk = sum(best_metrics['probabilities'] >= 0.7)
        medium_risk = sum((best_metrics['probabilities'] >= 0.4) & (best_metrics['probabilities'] < 0.7))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f5365c 0%, #f56036 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>High Risk</h3>
                <h1>{high_risk}</h1>
                <p>customers ({high_risk/total_customers:.1%})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>Medium Risk</h3>
                <h1>{medium_risk}</h1>
                <p>customers ({medium_risk/total_customers:.1%})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>Low Risk</h3>
                <h1>{total_customers - high_risk - medium_risk}</h1>
                <p>customers ({(total_customers - high_risk - medium_risk)/total_customers:.1%})</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Download Model Report")
        
        report_data = {
            'Model': st.session_state.best_model_name,
            'Accuracy': best_metrics['accuracy'],
            'Precision': best_metrics['precision'],
            'Recall': best_metrics['recall'],
            'F1 Score': best_metrics['f1'],
            'AUC-ROC': best_metrics['auc_roc'],
            'High Risk Customers': high_risk,
            'Medium Risk Customers': medium_risk,
            'Total Test Customers': total_customers
        }
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Performance Report",
            data=csv,
            file_name="model_performance_report.csv",
            mime="text/csv"
        )

with tabs[4]:
    st.header("Individual Customer Prediction")
    
    if st.session_state.best_model is None:
        st.warning("Please train models first in the Model Training tab.")
    else:
        st.subheader("Enter Customer Details")
        
        feature_inputs = {}
        categorical_cols = st.session_state.categorical_columns
        label_encoders = st.session_state.label_encoders
        original_data = st.session_state.original_data
        
        cols = st.columns(3)
        for idx, feature in enumerate(st.session_state.feature_columns):
            if feature in ['feature_sum', 'feature_mean', 'feature_std']:
                continue
            
            with cols[idx % 3]:
                if feature in categorical_cols and feature in label_encoders:
                    categories = list(label_encoders[feature].classes_)
                    selected_category = st.selectbox(
                        feature,
                        categories,
                        key=f"input_{feature}"
                    )
                    feature_inputs[feature] = label_encoders[feature].transform([selected_category])[0]
                elif st.session_state.cleaned_data is not None:
                    min_val = float(st.session_state.cleaned_data[feature].min())
                    max_val = float(st.session_state.cleaned_data[feature].max())
                    mean_val = float(st.session_state.cleaned_data[feature].mean())
                    
                    feature_inputs[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
        
        if st.button("Predict Churn", type="primary"):
            try:
                input_df = pd.DataFrame([feature_inputs])
                
                numeric_input_cols = [col for col in feature_inputs.keys() 
                                      if col not in st.session_state.categorical_columns]
                
                if len(numeric_input_cols) >= 2:
                    if 'feature_sum' in st.session_state.feature_columns:
                        input_df['feature_sum'] = input_df[numeric_input_cols].sum(axis=1)
                    if 'feature_mean' in st.session_state.feature_columns:
                        input_df['feature_mean'] = input_df[numeric_input_cols].mean(axis=1)
                    if 'feature_std' in st.session_state.feature_columns:
                        input_df['feature_std'] = input_df[numeric_input_cols].std(axis=1).fillna(0)
                
                for col in st.session_state.feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df[st.session_state.feature_columns]
                input_scaled = st.session_state.scaler.transform(input_df)
                
                probability = st.session_state.best_model.predict_proba(input_scaled)[0][1]
                prediction = st.session_state.best_model.predict(input_scaled)[0]
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                probability = 0.5
                prediction = 0
            
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if probability >= 0.7:
                    color = "#f5365c"
                    risk = "HIGH RISK"
                elif probability >= 0.4:
                    color = "#fb6340"
                    risk = "MEDIUM RISK"
                else:
                    color = "#2dce89"
                    risk = "LOW RISK"
                
                st.markdown(f"""
                <div style="background: {color}; padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                    <h2>{risk}</h2>
                    <h1>{probability:.1%}</h1>
                    <p>Churn Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 40], 'color': "#d4edda"},
                            {'range': [40, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#f8d7da"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Retention Recommendations")
            recommendations = get_retention_recommendations(
                probability,
                list(feature_inputs.values()),
                list(feature_inputs.keys())
            )
            
            for rec in recommendations:
                if "HIGH PRIORITY" in rec:
                    st.error(rec)
                elif "MEDIUM PRIORITY" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)

with tabs[5]:
    st.header("Customer Segmentation")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please preprocess data first in the Preprocessing tab.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_clusters = st.slider("Number of Segments", 2, 6, 3)
            
            if st.button("Perform Segmentation", type="primary"):
                X = st.session_state.cleaned_data[st.session_state.feature_columns]
                X_scaled = st.session_state.scaler.transform(X)
                
                clusters, kmeans = perform_clustering(X_scaled, n_clusters)
                st.session_state.cluster_labels = clusters
                
                st.success(f"Created {n_clusters} customer segments!")
        
        with col2:
            if st.session_state.cluster_labels is not None:
                df_with_clusters = st.session_state.cleaned_data.copy()
                df_with_clusters['Segment'] = st.session_state.cluster_labels
                
                segment_analysis = df_with_clusters.groupby('Segment').agg({
                    st.session_state.target_column: ['mean', 'count']
                }).round(3)
                segment_analysis.columns = ['Churn Rate', 'Customer Count']
                
                st.subheader("Segment Summary")
                st.dataframe(segment_analysis, use_container_width=True)
        
        if st.session_state.cluster_labels is not None:
            st.subheader("Segment Visualization")
            
            df_with_clusters = st.session_state.cleaned_data.copy()
            df_with_clusters['Segment'] = st.session_state.cluster_labels
            
            numeric_features = st.session_state.feature_columns[:min(3, len(st.session_state.feature_columns))]
            
            if len(numeric_features) >= 2:
                fig = px.scatter(
                    df_with_clusters,
                    x=numeric_features[0],
                    y=numeric_features[1],
                    color='Segment',
                    title="Customer Segments (2D View)",
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Segment Profiles")
            
            segment_profiles = df_with_clusters.groupby('Segment')[st.session_state.feature_columns].mean()
            
            fig = px.imshow(
                segment_profiles.T,
                labels=dict(x="Segment", y="Feature", color="Mean Value"),
                aspect='auto',
                color_continuous_scale='RdYlBu_r',
                title="Feature Heatmap by Segment"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Download Segmented Data")
            csv = df_with_clusters.to_csv(index=False)
            st.download_button(
                label="Download Segmented Customer Data",
                data=csv,
                file_name="segmented_customer_data.csv",
                mime="text/csv"
            )

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This Customer Churn Prediction Platform helps you:
- Upload and analyze customer data
- Train multiple ML models
- Identify at-risk customers
- Get actionable retention recommendations
- Segment customers for targeted marketing
""")

st.sidebar.markdown("### Quick Guide")
st.sidebar.markdown("""
1. **Upload** your CSV data
2. **Preprocess** and clean the data
3. **Train** machine learning models
4. **Analyze** model performance
5. **Predict** individual customer churn
6. **Segment** customers into groups
""")
