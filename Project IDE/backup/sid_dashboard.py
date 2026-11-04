import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="E-commerce Fraud Detection Dashboard",
    layout="wide"
)

st.title("📊 E-commerce Fraud Detection Dashboard")

# =============================================================
# LOAD DATA FROM HDFS
# =============================================================
@st.cache_data
def load_data():
    spark = SparkSession.builder.appName("EcommerceDashboard").getOrCreate()
    hdfs_path = "hdfs://192.168.56.101:8020/user/cloudera/ecommerce_dataset/ecommerce_transactions_with_products.csv"
    df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
    pdf = df.toPandas()
    pdf['transaction_time'] = pd.to_datetime(pdf['transaction_time'])
    return pdf

data = load_data()
st.success("✅ Data loaded successfully from HDFS!")

# =============================================================
# TOP METRICS
# =============================================================
total_transactions = len(data)
fraud_transactions = data['is_fraud'].sum()
fraud_percentage = (fraud_transactions / total_transactions) * 100
avg_transaction_amount = data['transaction_amount'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Cases", fraud_transactions)
col3.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
col4.metric("Avg Amount", f"${avg_transaction_amount:.2f}")

# =============================================================
# VISUALIZATION 1: Fraud by Payment Method
# =============================================================
st.subheader("💳 Fraud by Payment Method")

payment_summary = data.groupby('payment_method').agg({'is_fraud': ['count', 'sum']})
payment_summary.columns = ['total', 'fraud']
payment_summary['fraud_rate'] = (payment_summary['fraud'] / payment_summary['total']) * 100
payment_summary = payment_summary.reset_index()

fig1, ax1 = plt.subplots(figsize=(8, 4))
bars = ax1.bar(payment_summary['payment_method'], payment_summary['fraud_rate'], color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
ax1.set_ylabel('Fraud Rate (%)')
ax1.set_title('Fraud Rate by Payment Method')
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}%', ha='center', va='bottom')
st.pyplot(fig1)

# =============================================================
# VISUALIZATION 2: Monthly Fraud Trend
# =============================================================
st.subheader("📈 Monthly Fraud Trend")

data['month'] = data['transaction_time'].dt.to_period('M')
monthly_stats = data.groupby('month').agg({'transaction_id': 'count', 'is_fraud': 'sum'}).reset_index()
monthly_stats['month_str'] = monthly_stats['month'].astype(str)
monthly_stats['fraud_rate'] = (monthly_stats['is_fraud'] / monthly_stats['transaction_id']) * 100

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(monthly_stats['month_str'], monthly_stats['fraud_rate'], marker='o', color='red', linewidth=2)
ax2.set_xlabel('Month')
ax2.set_ylabel('Fraud Rate (%)')
ax2.set_title('Monthly Fraud Rate Trend')
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# =============================================================
# VISUALIZATION 3: Top 5 High-Risk Products
# =============================================================
st.subheader("🏆 Top 5 High-Risk Products")

product_fraud = data.groupby('product_name').agg({'transaction_id': 'count', 'is_fraud': 'sum'})
product_fraud['fraud_rate'] = (product_fraud['is_fraud'] / product_fraud['transaction_id']) * 100
top_products = product_fraud[product_fraud['transaction_id'] >= 10].nlargest(5, 'fraud_rate')

fig3, ax3 = plt.subplots(figsize=(8, 4))
bars = ax3.barh(top_products.index, top_products['fraud_rate'], color='#ff9999')
ax3.set_xlabel('Fraud Rate (%)')
ax3.set_title('Top 5 Products with Highest Fraud Rates')
for bar in bars:
    width = bar.get_width()
    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2., f'{width:.1f}%', ha='left', va='center')
st.pyplot(fig3)

# =============================================================
# 🧠 FRAUD PREDICTION MODEL
# =============================================================
st.markdown("---")
st.header("🤖 Fraud Detection Model")

# Encode categorical variables
model_data = data.copy()
categorical_cols = ['payment_method', 'product_name', 'location']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_data[col] = le.fit_transform(model_data[col].astype(str))
    encoders[col] = le

model_data['hour'] = model_data['transaction_time'].dt.hour

# Define features and target
X = model_data[['transaction_amount', 'payment_method', 'product_name', 'location', 'hour']]
y = model_data['is_fraud']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
st.success(f"✅ Fraud Detection Model Trained (Accuracy: {accuracy*100:.2f}%)")

# =============================================================
# USER INPUT FOR FRAUD PREDICTION
# =============================================================
st.subheader("🔍 Check if a Transaction is Fraudulent")

col1, col2, col3 = st.columns(3)
with col1:
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.5)
    payment_method = st.selectbox("Payment Method", data['payment_method'].unique())
with col2:
    product_name = st.selectbox("Product Name", data['product_name'].unique())
    location = st.selectbox("Location", data['location'].unique())
with col3:
    hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)

if st.button("Predict Fraud"):
    encoded_payment = encoders['payment_method'].transform([payment_method])[0]
    encoded_product = encoders['product_name'].transform([product_name])[0]
    encoded_location = encoders['location'].transform([location])[0]
    
    input_data = np.array([[transaction_amount, encoded_payment, encoded_product, encoded_location, hour]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 This transaction is **Fraudulent!** (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ This transaction is **Legitimate.** (Probability: {probability*100:.2f}%)")
