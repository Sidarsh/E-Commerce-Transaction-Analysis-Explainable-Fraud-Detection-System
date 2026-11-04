import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import numpy as np

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="E-commerce Fraud Dashboard",
    layout="wide"
)

st.title("📊 E-commerce Fraud Dashboard")

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

data = load_data()
st.success("✅ Data loaded successfully!")

# --------------------------
# Smaller Sample for Cleaner Graphs
# --------------------------
plot_data = data.sample(n=500, random_state=42)  # Reduced from 2000 to 500

# --------------------------
# Top Metrics - Clean and Simple
# --------------------------
total_transactions = len(data)
fraud_transactions = data['is_fraud'].sum()
fraud_percentage = (fraud_transactions / total_transactions) * 100
avg_transaction_amount = data['transaction_amount'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Cases", fraud_transactions)
col3.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
col4.metric("Avg Amount", f"${avg_transaction_amount:.2f}")

# --------------------------
# 1. Fraud by Payment Method - SIMPLIFIED
# --------------------------
st.subheader("💳 Fraud by Payment Method")

payment_summary = data.groupby('payment_method').agg({
    'is_fraud': ['count', 'sum']
}).round(2)
payment_summary.columns = ['total', 'fraud']
payment_summary['fraud_rate'] = (payment_summary['fraud'] / payment_summary['total']) * 100
payment_summary = payment_summary.reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 4))
bars = ax1.bar(payment_summary['payment_method'], payment_summary['fraud_rate'], 
               color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
ax1.set_ylabel('Fraud Rate (%)')
ax1.set_title('Fraud Rate by Payment Method')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom')

plt.xticks(rotation=0)
st.pyplot(fig1)

# --------------------------
# 2. Monthly Fraud Trend - CLEANER
# --------------------------
st.subheader("📈 Monthly Fraud Trend")

# Use monthly aggregation instead of daily
data['month'] = data['transaction_time'].dt.to_period('M')
monthly_stats = data.groupby('month').agg({
    'transaction_id': 'count',
    'is_fraud': 'sum'
}).reset_index()
monthly_stats['month_str'] = monthly_stats['month'].astype(str)
monthly_stats['fraud_rate'] = (monthly_stats['is_fraud'] / monthly_stats['transaction_id']) * 100

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(monthly_stats['month_str'], monthly_stats['fraud_rate'], 
         marker='o', linewidth=2, color='red', markersize=6)
ax2.set_ylabel('Fraud Rate (%)')
ax2.set_xlabel('Month')
ax2.set_title('Monthly Fraud Rate Trend')
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig2)

# --------------------------
# 3. Top 5 High-Risk Products
# --------------------------
st.subheader("🏆 Top 5 High-Risk Products")

product_fraud = data.groupby('product_name').agg({
    'transaction_id': 'count',
    'is_fraud': 'sum'
})
product_fraud['fraud_rate'] = (product_fraud['is_fraud'] / product_fraud['transaction_id']) * 100

# Get top 5 products with minimum 10 transactions
top_products = product_fraud[product_fraud['transaction_id'] >= 10].nlargest(5, 'fraud_rate')

fig3, ax3 = plt.subplots(figsize=(10, 4))
bars = ax3.barh(top_products.index, top_products['fraud_rate'], color='#ff9999')
ax3.set_xlabel('Fraud Rate (%)')
ax3.set_title('Top 5 Products with Highest Fraud Rates')
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
             f'{width:.1f}%', ha='left', va='center')

st.pyplot(fig3)

# --------------------------
# 4. Transaction Amount Analysis - SIMPLIFIED
# --------------------------
st.subheader("💰 Transaction Amount Analysis")

# Create amount categories for cleaner analysis
bins = [0, 100, 500, 1000, float('inf')]
labels = ['$0-100', '$100-500', '$500-1000', '$1000+']
data['amount_category'] = pd.cut(data['transaction_amount'], bins=bins, labels=labels)

amount_analysis = data.groupby('amount_category').agg({
    'transaction_id': 'count',
    'is_fraud': 'sum'
})
amount_analysis['fraud_rate'] = (amount_analysis['is_fraud'] / amount_analysis['transaction_id']) * 100

fig4, ax4 = plt.subplots(figsize=(10, 4))
bars = ax4.bar(amount_analysis.index.astype(str), amount_analysis['fraud_rate'], color='#66b3ff')
ax4.set_ylabel('Fraud Rate (%)')
ax4.set_xlabel('Transaction Amount Range')
ax4.set_title('Fraud Rate by Transaction Amount Range')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom')

st.pyplot(fig4)

# --------------------------
# 5. Top 5 High-Risk Locations
# --------------------------
st.subheader("🌍 Top 5 High-Risk Locations")

location_analysis = data.groupby('location').agg({
    'transaction_id': 'count',
    'is_fraud': 'sum'
})
location_analysis['fraud_rate'] = (location_analysis['is_fraud'] / location_analysis['transaction_id']) * 100

# Get top 5 locations with minimum 20 transactions
top_locations = location_analysis[location_analysis['transaction_id'] >= 20].nlargest(5, 'fraud_rate')

fig5, ax5 = plt.subplots(figsize=(10, 4))
bars = ax5.barh(top_locations.index, top_locations['fraud_rate'], color='#99ff99')
ax5.set_xlabel('Fraud Rate (%)')
ax5.set_title('Top 5 Locations with Highest Fraud Rates')
ax5.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax5.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
             f'{width:.1f}%', ha='left', va='center')

st.pyplot(fig5)

# --------------------------
# 6. Simple Hourly Pattern
# --------------------------
st.subheader("🕒 Fraud Pattern by Hour")

data['hour'] = data['transaction_time'].dt.hour
hourly_pattern = data.groupby('hour').agg({
    'transaction_id': 'count',
    'is_fraud': 'sum'
})
hourly_pattern['fraud_rate'] = (hourly_pattern['is_fraud'] / hourly_pattern['transaction_id']) * 100

fig6, ax6 = plt.subplots(figsize=(10, 4))
ax6.plot(hourly_pattern.index, hourly_pattern['fraud_rate'], 
         marker='o', linewidth=2, color='purple', markersize=6)
ax6.set_xlabel('Hour of Day')
ax6.set_ylabel('Fraud Rate (%)')
ax6.set_title('Fraud Rate by Hour of Day')
ax6.set_xticks(range(0, 24))
ax6.grid(True, alpha=0.3)
st.pyplot(fig6)

# --------------------------
# Simple Summary Tables
# --------------------------
st.subheader("📋 Quick Summary")

col7, col8 = st.columns(2)

with col7:
    st.write("**Payment Methods Summary:**")
    simple_payment = payment_summary[['payment_method', 'total', 'fraud_rate']].round(2)
    st.dataframe(simple_payment, use_container_width=True)

with col8:
    st.write("**High-Risk Products:**")
    simple_products = top_products[['transaction_id', 'fraud_rate']].round(2)
    simple_products.columns = ['Total Transactions', 'Fraud Rate %']
    st.dataframe(simple_products, use_container_width=True)

# --------------------------
# Simple Filters
# --------------------------
st.sidebar.header("🔍 Quick Filters")

# Simple payment method filter
selected_payment = st.sidebar.selectbox(
    "View by Payment Method:",
    ['All'] + list(data['payment_method'].unique())
)

# Simple amount range filter
selected_amount = st.sidebar.selectbox(
    "View by Amount Range:",
    ['All'] + list(data['amount_category'].cat.categories)
)

# Apply simple filters
filtered_view = data.copy()
if selected_payment != 'All':
    filtered_view = filtered_view[filtered_view['payment_method'] == selected_payment]
if selected_amount != 'All':
    filtered_view = filtered_view[filtered_view['amount_category'] == selected_amount]

st.sidebar.metric("Filtered Transactions", len(filtered_view))
st.sidebar.metric("Filtered Fraud Rate", 
                  f"{(filtered_view['is_fraud'].sum() / len(filtered_view) * 100):.1f}%")

# --------------------------
# Key Insights in Clean Format
# --------------------------
st.subheader("💡 Key Insights")

insight1, insight2, insight3 = st.columns(3)

highest_fraud_payment = payment_summary.loc[payment_summary['fraud_rate'].idxmax()]
highest_fraud_product = top_products.iloc[0]
highest_fraud_location = top_locations.iloc[0]

with insight1:
    st.info(f"**Highest Fraud Payment:** {highest_fraud_payment['payment_method']} ({highest_fraud_payment['fraud_rate']:.1f}%)")

with insight2:
    st.warning(f"**Riskiest Product:** {highest_fraud_product.name} ({highest_fraud_product['fraud_rate']:.1f}%)")

with insight3:
    st.error(f"**Riskiest Location:** {highest_fraud_location.name} ({highest_fraud_location['fraud_rate']:.1f}%)")