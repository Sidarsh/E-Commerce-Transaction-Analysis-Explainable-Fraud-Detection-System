import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="E-commerce Fraud Dashboard",
    layout="wide"
)

# --------------------------
# Load Data from HDFS
# --------------------------
@st.cache_data
def load_data():
    spark = SparkSession.builder.appName("EcommerceDashboard").getOrCreate()
    hdfs_path = "hdfs://192.168.56.101:8020/user/cloudera/dataset/ecommerce_transactions_dataset.csv"
    df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
    pdf = df.toPandas()
    pdf['transaction_time'] = pd.to_datetime(pdf['transaction_time'])
    return pdf

# --------------------------
# Train Model Function
# --------------------------
@st.cache_resource
def train_model(data):
    """Train a simple fraud detection model"""
    # Prepare features - ONLY the required ones
    features = data[['transaction_amount', 'payment_method', 'location', 'product_name']].copy()
    target = data['is_fraud']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['payment_method', 'location', 'product_name']
    
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        label_encoders[col] = le
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    
    return model, label_encoders

# --------------------------
# Prediction Function
# --------------------------
def predict_fraud(model, label_encoders, input_data):
    """Predict fraud for new transaction"""
    # Create feature DataFrame
    features = pd.DataFrame([input_data])
    
    # Encode categorical variables
    categorical_cols = ['payment_method', 'location', 'product_name']
    for col in categorical_cols:
        le = label_encoders[col]
        # Handle unseen labels
        if input_data[col] in le.classes_:
            features[col] = le.transform([input_data[col]])[0]
        else:
            features[col] = -1  # Assign -1 for unseen labels
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

# --------------------------
# Dashboard Page
# --------------------------
def show_dashboard(data):
    st.title("📊 E-commerce Fraud Dashboard")
    st.success("✅ Data loaded successfully!")
    
    # Prediction button at top
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔮 Predict Fraud for New Transaction", use_container_width=True, type="primary"):
            st.session_state.show_prediction = True
            st.rerun()
    st.markdown("---")
    
    plot_data = data.sample(n=500, random_state=42)

    # --------------------------
    # Top Metrics
    # --------------------------
    total_transactions = len(data)
    fraud_transactions = data['is_fraud'].sum()
    fraud_percentage = (fraud_transactions / total_transactions) * 100
    avg_transaction_amount = data['transaction_amount'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Cases", fraud_transactions)
    col3.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
    col4.metric("Avg Amount", f"₹{avg_transaction_amount:.2f}")

    # --------------------------
    # Price Anomaly Detection
    # --------------------------
    st.subheader("⚠ Price Anomalies")
    avg_price_dict = data.groupby('product_name')['transaction_amount'].mean().to_dict()
    data['price_ratio'] = data.apply(lambda row: row['transaction_amount']/avg_price_dict.get(row['product_name'],1), axis=1)
    price_anomalies = data[(data['price_ratio'] < 0.1) | (data['price_ratio'] > 2.0)]
    if not price_anomalies.empty:
        st.warning(f"Detected {len(price_anomalies)} suspicious transactions based on price deviation")
        st.dataframe(price_anomalies[['transaction_id','product_name','transaction_amount','price_ratio']], use_container_width=True)
    else:
        st.success("No significant price anomalies detected")
    
    # --------------------------
    # Fraud by Payment Method
    # --------------------------
    st.subheader("💳 Fraud by Payment Method")
    payment_summary = data.groupby('payment_method').agg({'is_fraud':['count','sum']}).round(2)
    payment_summary.columns = ['total','fraud']
    payment_summary['fraud_rate'] = (payment_summary['fraud'] / payment_summary['total']) * 100
    payment_summary = payment_summary.reset_index()

    fig1, ax1 = plt.subplots(figsize=(10,4))
    bars = ax1.bar(payment_summary['payment_method'], payment_summary['fraud_rate'], color=['#ff6b6b','#4ecdc4','#45b7d1','#96ceb4'])
    ax1.set_ylabel('Fraud Rate (%)')
    ax1.set_title('Fraud Rate by Payment Method')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.1, f"{bar.get_height():.1f}%", ha='center', va='bottom')
    plt.xticks(rotation=0)
    st.pyplot(fig1)

    # --------------------------
    # Monthly Fraud Trend
    # --------------------------
    st.subheader("📈 Monthly Fraud Trend")
    data['month'] = data['transaction_time'].dt.to_period('M')
    monthly_stats = data.groupby('month').agg({'transaction_id':'count','is_fraud':'sum'}).reset_index()
    monthly_stats['month_str'] = monthly_stats['month'].astype(str)
    monthly_stats['fraud_rate'] = (monthly_stats['is_fraud']/monthly_stats['transaction_id'])*100

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(monthly_stats['month_str'], monthly_stats['fraud_rate'], marker='o', linewidth=2, color='red', markersize=6)
    ax2.set_ylabel('Fraud Rate (%)')
    ax2.set_xlabel('Month')
    ax2.set_title('Monthly Fraud Rate Trend')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # --------------------------
    # Top 5 High-Risk Products
    # --------------------------
    st.subheader("🏆 Top 5 High-Risk Products")
    product_fraud = data.groupby('product_name').agg({'transaction_id':'count','is_fraud':'sum'})
    product_fraud['fraud_rate'] = (product_fraud['is_fraud']/product_fraud['transaction_id'])*100
    top_products = product_fraud[product_fraud['transaction_id']>=10].nlargest(5,'fraud_rate')

    fig3, ax3 = plt.subplots(figsize=(10,4))
    bars = ax3.barh(top_products.index, top_products['fraud_rate'], color='#ff9999')
    ax3.set_xlabel('Fraud Rate (%)')
    ax3.set_title('Top 5 Products with Highest Fraud Rates')
    ax3.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        ax3.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2., f"{bar.get_width():.1f}%", ha='left', va='center')
    st.pyplot(fig3)

    # --------------------------
    # Continue your existing code...
    # (Transaction Amount Analysis, Locations, Hourly Pattern, Summary, Filters, Key Insights)
    # You can keep the existing blocks as-is.

# --------------------------
# Prediction Page
# --------------------------
def show_prediction_page(data):
    st.title("🔮 Fraud Prediction")
    
    # Back to dashboard button
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("📊 Back to Dashboard", use_container_width=True, type="secondary"):
            st.session_state.show_prediction = False
            st.rerun()
    st.markdown("---")
    
    st.markdown("### Enter Transaction Details to Predict Fraud")
    st.info("🔍 *Prediction is based on:* Payment Method, Product Name, Transaction Amount, and Location")
    
    # Train model
    model, label_encoders = train_model(data)
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            product_name = st.selectbox("Product Name*", options=data['product_name'].unique())
            transaction_amount = st.number_input("Transaction Amount (₹)*", min_value=0.0, value=25000.0, step=1000.0)
        with col2:
            payment_method = st.selectbox("Payment Method*", options=data['payment_method'].unique())
            location = st.selectbox("Location*", options=data['location'].unique())
        submitted = st.form_submit_button("Predict Fraud", use_container_width=True)
    
    if submitted:
        input_data = {'transaction_amount': transaction_amount,
                      'payment_method': payment_method,
                      'location': location,
                      'product_name': product_name}
        prediction, probability = predict_fraud(model, label_encoders, input_data)
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        if prediction==1:
            st.error(f"🚨 FRAUD DETECTED! Probability: {probability:.2%}")
        else:
            st.success(f"✅ LEGITIMATE TRANSACTION. Probability: {probability:.2%}")

# --------------------------
# Main App
# --------------------------
def main():
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    data = load_data()
    if st.session_state.show_prediction:
        show_prediction_page(data)
    else:
        show_dashboard(data)

if __name__ == "__main__":
    main()
