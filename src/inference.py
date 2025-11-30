import pandas as pd
import joblib
import xgboost as xgb
from src.config import MODEL_PATH, ENCODER_PATH, DATA_RAW
from src.features import preprocess_data

# Global cache for artifacts to avoid reloading on every request
_MODEL = None
_SELLER_STATS = None
_GEO_DATA = None

def load_artifacts():
    """Loads model, seller stats, and geo data."""
    global _MODEL, _SELLER_STATS, _GEO_DATA
    
    if _MODEL is None:
        print(f"Loading model from {MODEL_PATH}...")
        _MODEL = joblib.load(MODEL_PATH)
    
    if _SELLER_STATS is None:
        print(f"Loading seller stats from {ENCODER_PATH}...")
        _SELLER_STATS = joblib.load(ENCODER_PATH)
        
    if _GEO_DATA is None:
        print("Loading geo data...")
        geo_path = DATA_RAW / 'olist_geolocation_dataset.csv'
        _GEO_DATA = pd.read_csv(geo_path)
        
    return _MODEL, _SELLER_STATS, _GEO_DATA

def predict(data):
    """
    Makes predictions for a single order or a batch of orders.
    
    Args:
        data (dict or list of dicts): Input data containing order details.
            Required fields: 
            - seller_zip_code_prefix (or s_lat, s_lng)
            - customer_zip_code_prefix (or c_lat, c_lng)
            - seller_id
            - product_weight_g
            - freight_value
            - price
            
    Returns:
        list: List of dictionaries containing 'order_id' (if present) and 'delay_risk_probability'.
    """
    model, seller_stats, geo_data = load_artifacts()
    
    # Convert input to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
        
    # Preprocess
    # We pass the pre-computed seller_stats so it uses historical data instead of computing from the input
    df_processed, _ = preprocess_data(df, geo_data=geo_data, seller_stats=seller_stats)
    
    if df_processed.empty:
        return [{"error": "Could not preprocess data. Check missing fields or invalid zip codes."}]
    
    # Select features
    features = ['distance_km', 'seller_late_rate', 'product_weight_g', 'freight_value', 'price']
    
    # Ensure all features exist (handle missing columns if preprocessing failed to create them)
    missing_cols = [col for col in features if col not in df_processed.columns]
    if missing_cols:
         return [{"error": f"Missing features after preprocessing: {missing_cols}"}]

    X = df_processed[features]
    
    # Predict
    probs = model.predict_proba(X)[:, 1]
    
    # Format response
    results = []
    for i, prob in enumerate(probs):
        result = {
            "delay_risk_probability": float(prob),
            "is_high_risk": bool(prob > 0.5) # Default threshold
        }
        if 'order_id' in df_processed.iloc[i]:
            result['order_id'] = df_processed.iloc[i]['order_id']
        results.append(result)
        
    return results

if __name__ == "__main__":
    # Test run
    sample_data = {
        "seller_id": "e481f51cbdc54678b7cc49136f2d6af7", # Example ID
        "seller_zip_code_prefix": 14409,
        "customer_zip_code_prefix": 14409,
        "product_weight_g": 500,
        "freight_value": 15.0,
        "price": 50.0
    }
    # Note: This will fail if artifacts don't exist yet.
    # print(predict(sample_data))
    pass
