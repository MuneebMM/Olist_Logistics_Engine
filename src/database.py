import sqlite3
import pandas as pd
from pathlib import Path
from src.config import DB_PATH, DATA_RAW

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # Ensure the directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def load_raw_data():
    """Loads raw CSV files into the SQLite database."""
    conn = get_db_connection()
    
    # Map CSV filenames to table names
    files = {
        'olist_orders_dataset.csv': 'orders',
        'olist_order_items_dataset.csv': 'order_items',
        'olist_products_dataset.csv': 'products',
        'olist_sellers_dataset.csv': 'sellers',
        'olist_customers_dataset.csv': 'customers',
        'olist_order_reviews_dataset.csv': 'order_reviews',
        'olist_geolocation_dataset.csv': 'geolocation'
    }
    
    print("Loading raw data into SQLite...")
    for csv_file, table_name in files.items():
        file_path = DATA_RAW / csv_file
        if file_path.exists():
            print(f"Loading {table_name}...")
            df = pd.read_csv(file_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        else:
            print(f"Warning: {file_path} not found.")
            
    conn.close()
    print("Raw data loaded successfully.")

def create_master_table():
    """Creates the master_analytics_table in the SQLite database."""
    query = """
    CREATE TABLE IF NOT EXISTS master_analytics_table AS
    SELECT
        -- 1. Order Basics
        o.order_id,
        o.customer_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        oi.seller_id,

        -- 2. Target Variable (The "Is_Late" Flag)
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
            ELSE 0 
        END as is_late,
        
        -- 3. Economic Data
        oi.price,
        oi.freight_value,
        
        -- 4. Product Features
        COALESCE(p.product_weight_g, (SELECT AVG(product_weight_g) FROM products)) as product_weight_g,
        p.product_category_name,
        
        -- 5. Location Data
        s.seller_zip_code_prefix,
        s.seller_state,
        c.customer_zip_code_prefix,
        c.customer_state,
        
        -- 6. Success Metric
        r.review_score

    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id

    WHERE 
        o.order_status = 'delivered' 
        AND o.order_delivered_customer_date IS NOT NULL;
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS master_analytics_table") # Re-create to be safe
    cursor.execute(query)
    conn.commit()
    conn.close()
    print("Master analytics table created successfully.")

def load_data(query="SELECT * FROM master_analytics_table"):
    """Loads data from the database using a SQL query."""
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df
