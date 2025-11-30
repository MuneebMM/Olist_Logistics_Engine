import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.inference import predict
from src.database import load_data

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Olist 360° Logistics Engine", layout="wide", page_icon="🚚")

# Custom CSS for that "Professional" look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING (Cached for Speed) ---
@st.cache_resource
def ensure_database_exists():
    """Checks if DB exists and has the master table. If not, rebuilds it."""
    from src.config import DB_PATH
    from src.database import load_raw_data, create_master_table, get_db_connection
    import sqlite3

    rebuild = False
    
    if not DB_PATH.exists():
        rebuild = True
    else:
        # Check if table exists
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='master_analytics_table';")
            if cursor.fetchone() is None:
                rebuild = True
            conn.close()
        except Exception:
            rebuild = True

    if rebuild:
        with st.spinner("⚠️ Database/Table not found. Building from raw data... (This may take a minute)"):
            try:
                load_raw_data()
                create_master_table()
                st.success("Database built successfully!")
            except Exception as e:
                st.error(f"Failed to build database: {e}")
                st.stop()

@st.cache_data
def load_dashboard_data():
    ensure_database_exists()
    try:
        # Load data from our SQLite DB
        df = load_data("SELECT * FROM master_analytics_table")
        return df
    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}")
        return pd.DataFrame()

# Load Data
df = load_dashboard_data()

if not df.empty:
    # --- 3. SIDEBAR: THE SIMULATOR ---
    st.sidebar.header("🕹️ Logistics Simulator")
    st.sidebar.write("Predict delay risk for a new order:")

    # Inputs matching our model features
    input_seller_id = st.sidebar.text_input("Seller ID", value="e481f51cbdc54678b7cc49136f2d6af7")
    input_seller_zip = st.sidebar.number_input("Seller Zip Code", 0, 99999, 14409)
    input_cust_zip = st.sidebar.number_input("Customer Zip Code", 0, 99999, 14409)
    input_dist = st.sidebar.slider("Distance (km) [Est.]", 0, 4000, 500) # Note: Real app calculates this from zips
    input_weight = st.sidebar.slider("Product Weight (g)", 0, 30000, 1000)
    input_freight = st.sidebar.number_input("Freight Cost (R$)", 10.0, 500.0, 50.0)
    input_price = st.sidebar.number_input("Price (R$)", 10.0, 5000.0, 100.0)

    # Prediction Logic using our actual inference pipeline
    if st.sidebar.button("Predict Risk"):
        input_data = {
            "seller_id": input_seller_id,
            "seller_zip_code_prefix": input_seller_zip,
            "customer_zip_code_prefix": input_cust_zip,
            "product_weight_g": input_weight,
            "freight_value": input_freight,
            "price": input_price,
            # We can pass distance if our backend supports overriding it, 
            # otherwise it will be recalculated from zips. 
            # For this UI, let's assume the backend calculates it.
        }
        
        with st.spinner("Calculating..."):
            try:
                results = predict(input_data)
                result = results[0]
                
                if 'error' in result:
                    st.sidebar.error(f"Error: {result['error']}")
                else:
                    risk_prob = result['delay_risk_probability']
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("Prediction Result")
                    if risk_prob > 0.5: 
                        st.sidebar.error(f"⚠️ HIGH RISK: {risk_prob:.1%} probability of delay")
                    else:
                        st.sidebar.success(f"✅ ON TRACK: {risk_prob:.1%} probability of delay")
            except Exception as e:
                st.sidebar.error(f"Prediction failed: {e}")

    # --- 4. MAIN DASHBOARD ---
    st.title("🚚 Olist 360° Logistics Command Center")
    st.markdown("Real-time monitoring of delivery bottlenecks and seller performance.")

    # Row A: KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_orders = len(df)
    late_rate = df['is_late'].mean()
    avg_score = df['review_score'].mean()
    avg_freight = df['freight_value'].mean()
    
    with kpi1:
        st.metric("Total Orders Processed", f"{total_orders:,}")
    with kpi2:
        st.metric("Global Late Rate", f"{late_rate:.1%}")
    with kpi3:
        st.metric("Avg CSAT Score", f"{avg_score:.2f} ⭐")
    with kpi4:
        st.metric("Avg Freight Value", f"R$ {avg_freight:.2f}")

    st.markdown("---")

    # Row B: The Logistics Funnel & Map
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("⏳ The Logistics Funnel (Mock Data)")
        # Creating a Funnel Data structure (Mock for visuals as per request)
        funnel_data = dict(
            number=[100, 95, 88], 
            stage=["Order Approved", "Handed to Carrier", "Delivered On Time"]
        )
        fig_funnel = px.funnel(funnel_data, x='number', y='stage', title="Order Drop-off Rates")
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_right:
        st.subheader("🗺️ Risk Heatmap")
        st.info("Visualizing 'Bad Routes' (Mock Data).")
        # Simple scatter map placeholder
        map_data = pd.DataFrame({
            'lat': np.random.uniform(-30, -5, 100),
            'lon': np.random.uniform(-60, -35, 100)
        })
        st.map(map_data)

    # Row C: Seller Performance Analysis
    st.subheader("📉 Worst Performing Categories (Delay Drivers)")
    
    if 'product_category_name' in df.columns:
        bad_cats = df[df['is_late']==1]['product_category_name'].value_counts().head(10).reset_index()
        bad_cats.columns = ['Category', 'Late Count']
        
        fig_bar = px.bar(bad_cats, x='Late Count', y='Category', orientation='h', color='Late Count', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Product category data not available.")

else:
    st.warning("Database is empty or could not be loaded.")
