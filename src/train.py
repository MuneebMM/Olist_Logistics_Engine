import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.config import MODEL_PATH, ENCODER_PATH, DATA_RAW
from src.database import load_data
from src.features import preprocess_data

import mlflow
import mlflow.xgboost

def train_model():
    """Trains the XGBoost model and saves artifacts."""
    mlflow.set_experiment("olist_logistics_delay_prediction")
    
    print("Loading data...")
    # Load orders data from DB (Master Table)
    df = load_data()
    
    # Load Geo Data for feature engineering
    geo_path = DATA_RAW / 'olist_geolocation_dataset.csv'
    geo_data = pd.read_csv(geo_path)
    
    print("Preprocessing data...")
    # Preprocess and compute seller stats
    df_clean, seller_stats = preprocess_data(df, geo_data=geo_data)
    
    # Define features and target
    features = ['distance_km', 'seller_late_rate', 'product_weight_g', 'freight_value', 'price']
    target = 'is_late'
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model...")
    
    # Hyperparameters
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "scale_pos_weight": 10,
        "eval_metric": 'logloss',
        "use_label_encoder": False,
        "random_state": 42
    }
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train XGBoost
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_probs)
        
        print("--- Model Performance ---")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save Model locally (legacy support for app)
        print(f"Saving model to {MODEL_PATH}...")
        joblib.dump(model, MODEL_PATH)
        
        # Save Seller Stats (as our "Encoder" / Artifact for inference)
        print(f"Saving seller stats to {ENCODER_PATH}...")
        joblib.dump(seller_stats, ENCODER_PATH)
        
        # Log seller stats as artifact
        mlflow.log_artifact(str(ENCODER_PATH), "artifacts")
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
