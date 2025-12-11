import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get('DATABASE_URL')

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()


class PredictionHistory(Base):
    """Store individual customer predictions"""
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(100), nullable=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    feature_values = Column(JSON, nullable=True)
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(50), nullable=True)


class ModelVersion(Base):
    """Store trained model versions and their performance"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)


class ABExperiment(Base):
    """Store A/B testing experiments for retention strategies"""
    __tablename__ = 'ab_experiments'
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    status = Column(String(20), default='draft')
    control_strategy = Column(Text, nullable=True)
    treatment_strategy = Column(Text, nullable=True)
    target_segment = Column(String(100), nullable=True)
    sample_size = Column(Integer, nullable=True)


class ABExperimentResult(Base):
    """Store results for A/B experiments"""
    __tablename__ = 'ab_experiment_results'
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    group_type = Column(String(20), nullable=False)
    customer_count = Column(Integer, nullable=True)
    churn_rate = Column(Float, nullable=True)
    retention_rate = Column(Float, nullable=True)
    avg_clv = Column(Float, nullable=True)


class DataDriftLog(Base):
    """Track data drift detection results"""
    __tablename__ = 'data_drift_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    check_date = Column(DateTime, default=datetime.utcnow)
    feature_name = Column(String(100), nullable=False)
    drift_score = Column(Float, nullable=True)
    drift_detected = Column(Boolean, default=False)
    reference_mean = Column(Float, nullable=True)
    current_mean = Column(Float, nullable=True)
    reference_std = Column(Float, nullable=True)
    current_std = Column(Float, nullable=True)


class CLVPrediction(Base):
    """Store customer lifetime value predictions"""
    __tablename__ = 'clv_predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(100), nullable=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    predicted_clv = Column(Float, nullable=False)
    clv_segment = Column(String(20), nullable=True)
    churn_probability = Column(Float, nullable=True)
    expected_tenure = Column(Float, nullable=True)


def init_db():
    """Initialize database tables"""
    if engine:
        Base.metadata.create_all(bind=engine)
        return True
    return False


def get_db():
    """Get database session"""
    if SessionLocal:
        db = SessionLocal()
        try:
            return db
        except:
            db.close()
            return None
    return None
