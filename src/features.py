import pandas as pd
import numpy as np

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points on the earth."""
    r = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c

def compute_seller_stats(df):
    """Computes seller statistics (late rate)."""
    # "How often has this seller been late in the past?"
    # We'll use a simple global average for this demo to avoid data leakage in a complex time-series split
    # Note: In a real production system, this should be an expanding mean or calculated on a holdout set.
    seller_stats = df.groupby('seller_id')['is_late'].agg(['mean', 'count']).reset_index()
    seller_stats.columns = ['seller_id', 'seller_late_rate', 'seller_order_count']
    return seller_stats

def preprocess_data(df, geo_data=None, seller_stats=None):
    """
    Applies all feature engineering steps.
    
    Args:
        df: The input dataframe (orders merged with items, sellers, customers).
        geo_data: Optional dataframe containing geolocation data. If None, it assumes lat/lng columns might already be present or skips geo features if they can't be computed.
        seller_stats: Optional dataframe containing pre-computed seller stats. If None, they are computed from the input df.
    
    Returns:
        df: The dataframe with new features.
        seller_stats: The computed seller stats (useful for saving as artifact).
    """
    
    # 1. Geo Features
    if geo_data is not None:
        # Geo data has duplicates (many lat/lngs for one zip). We take the mean.
        geo_agg = geo_data.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
        
        # Merge Seller Coords
        if 'seller_zip_code_prefix' in df.columns and 's_lat' not in df.columns:
            df = df.merge(geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
            df.rename(columns={'geolocation_lat': 's_lat', 'geolocation_lng': 's_lng'}, inplace=True)
            df.drop(columns='geolocation_zip_code_prefix', inplace=True)
            
        # Merge Customer Coords
        if 'customer_zip_code_prefix' in df.columns and 'c_lat' not in df.columns:
            df = df.merge(geo_agg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
            df.rename(columns={'geolocation_lat': 'c_lat', 'geolocation_lng': 'c_lng'}, inplace=True)
            df.drop(columns='geolocation_zip_code_prefix', inplace=True)

    # Calculate Distance
    if 's_lat' in df.columns and 'c_lat' in df.columns:
        df['distance_km'] = calculate_haversine_distance(df['s_lat'], df['s_lng'], df['c_lat'], df['c_lng'])
    
    # 2. Seller Stats
    if seller_stats is None:
        seller_stats = compute_seller_stats(df)
    
    # Merge back
    if 'seller_late_rate' not in df.columns:
        df = df.merge(seller_stats[['seller_id', 'seller_late_rate']], on='seller_id', how='left')
        
    # Fill missing seller_late_rate with global mean (for new sellers)
    if 'seller_late_rate' in df.columns:
        global_mean_late_rate = seller_stats['seller_late_rate'].mean()
        df['seller_late_rate'] = df['seller_late_rate'].fillna(global_mean_late_rate)

    # 3. Fill NaNs
    if 'product_weight_g' in df.columns:
        df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].mean())
        
    # Drop rows where essential features are missing (only distance_km now)
    df = df.dropna(subset=['distance_km'])
    
    return df, seller_stats
