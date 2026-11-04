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
# Load Data from HDFSpyr
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
# Enhanced Feature Engineering
# --------------------------
def create_enhanced_features(data):
    """Create additional features for better fraud detection"""
    df = data.copy()
    
    # Amount-based risk flags
    df['is_very_low_amount'] = (df['transaction_amount'] <= 100).astype(int)
    df['is_very_high_amount'] = (df['transaction_amount'] > 50000).astype(int)
    df['is_suspicious_amount'] = ((df['transaction_amount'] <= 10) | 
                                 (df['transaction_amount'] % 1 != 0) |  # Decimal amounts
                                 (df['transaction_amount'] == 0)).astype(int)
    
    # High-risk product categories
    high_value_products = ['Smartphone', 'Laptop', 'Television', 'Camera', 'Gaming Console']
    df['is_high_value_product'] = df['product_name'].isin(high_value_products).astype(int)
    
    # High-risk payment methods
    high_risk_payments = ['Cash on Delivery']
    df['is_high_risk_payment'] = df['payment_method'].isin(high_risk_payments).astype(int)
    
    # High-risk locations
    high_risk_locations = ['Delhi', 'Mumbai', 'Kolkata']
    df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
    
    return df

# --------------------------
# Train Enhanced Model
# --------------------------
@st.cache_resource
def train_enhanced_model(data):
    """Train an enhanced fraud detection model"""
    # Create enhanced features
    enhanced_data = create_enhanced_features(data)
    
    # Prepare features - Core features + enhanced features
    core_features = ['transaction_amount', 'payment_method', 'location', 'product_name']
    enhanced_features = ['is_very_low_amount', 'is_very_high_amount', 'is_suspicious_amount',
                        'is_high_value_product', 'is_high_risk_payment', 'is_high_risk_location']
    
    features = enhanced_data[core_features + enhanced_features].copy()
    target = enhanced_data['is_fraud']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['payment_method', 'location', 'product_name']
    
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        label_encoders[col] = le
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(features, target)
    
    return model, label_encoders

# --------------------------
# Enhanced Prediction Function with High-Risk Combinations
# --------------------------
def predict_fraud_enhanced(model, label_encoders, input_data):
    """Enhanced fraud prediction with high-risk combination detection"""
    # Create enhanced features for the input
    features_dict = input_data.copy()
    
    # Amount-based risk flags
    features_dict['is_very_low_amount'] = 1 if input_data['transaction_amount'] <= 100 else 0
    features_dict['is_very_high_amount'] = 1 if input_data['transaction_amount'] > 50000 else 0
    features_dict['is_suspicious_amount'] = 1 if (input_data['transaction_amount'] <= 10 or 
                                                 input_data['transaction_amount'] % 1 != 0 or
                                                 input_data['transaction_amount'] == 0) else 0
    
    # Product risk
    high_value_products = ['Smartphone', 'Laptop', 'Television', 'Camera', 'Gaming Console']
    features_dict['is_high_value_product'] = 1 if input_data['product_name'] in high_value_products else 0
    
    # Payment risk
    high_risk_payments = ['Cash on Delivery']
    features_dict['is_high_risk_payment'] = 1 if input_data['payment_method'] in high_risk_payments else 0
    
    # Location risk
    high_risk_locations = ['Delhi', 'Mumbai', 'Kolkata']
    features_dict['is_high_risk_location'] = 1 if input_data['location'] in high_risk_locations else 0
    
    # Create feature DataFrame
    core_features = ['transaction_amount', 'payment_method', 'location', 'product_name']
    enhanced_features = ['is_very_low_amount', 'is_very_high_amount', 'is_suspicious_amount',
                        'is_high_value_product', 'is_high_risk_payment', 'is_high_risk_location']
    
    all_features = core_features + enhanced_features
    features_df = pd.DataFrame([{k: features_dict[k] for k in all_features}])
    
    # Encode categorical variables
    categorical_cols = ['payment_method', 'location', 'product_name']
    for col in categorical_cols:
        le = label_encoders[col]
        if input_data[col] in le.classes_:
            features_df[col] = le.transform([input_data[col]])[0]
        else:
            features_df[col] = -1
    
    # Make initial prediction
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]
    
    # CRITICAL FIX: High-risk combination overrides
    high_value_threshold = 50000  # ₹50,000
    
    # Rule 1: High-value product + Cash on Delivery + High amount
    if (features_dict['is_high_value_product'] and 
        features_dict['is_high_risk_payment'] and 
        input_data['transaction_amount'] > high_value_threshold):
        prediction = 1
        probability = max(probability, 0.90)  # Very high probability
        st.session_state.risk_override_reason = "High-value product with Cash on Delivery and large amount"
    
    # Rule 2: High-value + Cash on Delivery + High-risk location
    elif (features_dict['is_high_value_product'] and 
          features_dict['is_high_risk_payment'] and 
          features_dict['is_high_risk_location']):
        prediction = 1
        probability = max(probability, 0.85)
        st.session_state.risk_override_reason = "High-value product with Cash on Delivery in high-risk location"
    
    # Rule 3: Very low amount for high-value products
    elif (input_data['transaction_amount'] <= 1 and 
          features_dict['is_high_value_product']):
        prediction = 1
        probability = max(probability, 0.95)
        st.session_state.risk_override_reason = "Extremely low amount for high-value product"
    
    # Rule 4: High amount + Cash on Delivery
    elif (input_data['transaction_amount'] > high_value_threshold and 
          features_dict['is_high_risk_payment']):
        prediction = 1
        probability = max(probability, 0.80)
        st.session_state.risk_override_reason = "Large amount with Cash on Delivery"
    
    # Rule 5: Multiple risk factors combined
    risk_factors_count = (features_dict['is_high_value_product'] + 
                         features_dict['is_high_risk_payment'] + 
                         features_dict['is_high_risk_location'] +
                         features_dict['is_very_high_amount'])
    
    if risk_factors_count >= 3:  # If 3 or more risk factors
        prediction = 1
        probability = max(probability, 0.75)
        st.session_state.risk_override_reason = f"Multiple risk factors detected ({risk_factors_count})"
    
    else:
        st.session_state.risk_override_reason = None
    
    return prediction, probability

# --------------------------
# Dashboard Page
# --------------------------
def show_dashboard(data):
    st.title("📊 E-commerce Fraud Dashboard")
    st.success("✅ Data loaded successfully!")
    
    # Prediction button at top - Centered and prominent
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Predict Fraud for New Transaction", use_container_width=True, type="primary"):
            st.session_state.show_prediction = True
            st.rerun()
    st.markdown("---")
    
    plot_data = data.sample(n=500, random_state=42)

    # Top Metrics - Clean and Simple
    total_transactions = len(data)
    fraud_transactions = data['is_fraud'].sum()
    fraud_percentage = (fraud_transactions / total_transactions) * 100
    avg_transaction_amount = data['transaction_amount'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Cases", fraud_transactions)
    col3.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
    col4.metric("Avg Amount", f"₹{avg_transaction_amount:.2f}")

    # 1. Fraud by Payment Method - SIMPLIFIED
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

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.xticks(rotation=0)
    st.pyplot(fig1)

    # 2. Monthly Fraud Trend - CLEANER
    st.subheader("📈 Monthly Fraud Trend")

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

    # 3. Top 5 High-Risk Products
    st.subheader("🏆 Top 5 High-Risk Products")

    product_fraud = data.groupby('product_name').agg({
        'transaction_id': 'count',
        'is_fraud': 'sum'
    })
    product_fraud['fraud_rate'] = (product_fraud['is_fraud'] / product_fraud['transaction_id']) * 100

    top_products = product_fraud[product_fraud['transaction_id'] >= 10].nlargest(5, 'fraud_rate')

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    bars = ax3.barh(top_products.index, top_products['fraud_rate'], color='#ff9999')
    ax3.set_xlabel('Fraud Rate (%)')
    ax3.set_title('Top 5 Products with Highest Fraud Rates')
    ax3.grid(axis='x', alpha=0.3)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                 f'{width:.1f}%', ha='left', va='center')

    st.pyplot(fig3)

    # 4. Transaction Amount Analysis - COMMENTED OUT
    # st.subheader("💰 Transaction Amount Analysis")

    # # Updated bins for Indian Rupees
    # bins = [0, 10000, 50000, 100000, float('inf')]
    # labels = ['₹0-10K', '₹10K-50K', '₹50K-1L', '₹1L+']
    # data['amount_category'] = pd.cut(data['transaction_amount'], bins=bins, labels=labels)

    # amount_analysis = data.groupby('amount_category').agg({
    #     'transaction_id': 'count',
    #     'is_fraud': 'sum'
    # })
    # amount_analysis['fraud_rate'] = (amount_analysis['is_fraud'] / amount_analysis['transaction_id']) * 100

    # fig4, ax4 = plt.subplots(figsize=(10, 4))
    # bars = ax4.bar(amount_analysis.index.astype(str), amount_analysis['fraud_rate'], color='#66b3ff')
    # ax4.set_ylabel('Fraud Rate (%)')
    # ax4.set_xlabel('Transaction Amount Range')
    # ax4.set_title('Fraud Rate by Transaction Amount Range')
    # ax4.grid(axis='y', alpha=0.3)

    # for bar in bars:
    #     height = bar.get_height()
    #     ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
    #              f'{height:.1f}%', ha='center', va='bottom')

    # st.pyplot(fig4)

    # 5. Top 5 High-Risk Locations
    st.subheader("🌍 Top 5 High-Risk Locations")

    location_analysis = data.groupby('location').agg({
        'transaction_id': 'count',
        'is_fraud': 'sum'
    })
    location_analysis['fraud_rate'] = (location_analysis['is_fraud'] / location_analysis['transaction_id']) * 100

    top_locations = location_analysis[location_analysis['transaction_id'] >= 20].nlargest(5, 'fraud_rate')

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    bars = ax5.barh(top_locations.index, top_locations['fraud_rate'], color='#99ff99')
    ax5.set_xlabel('Fraud Rate (%)')
    ax5.set_title('Top 5 Locations with Highest Fraud Rates')
    ax5.grid(axis='x', alpha=0.3)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax5.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                 f'{width:.1f}%', ha='left', va='center')

    st.pyplot(fig5)

    # 6. Simple Hourly Pattern
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

    # Simple Summary Tables
    st.subheader("📋 Quick Summary")

    col7, col8 = st.columns(2)

    with col7:
        st.write("Payment Methods Summary:")
        simple_payment = payment_summary[['payment_method', 'total', 'fraud_rate']].round(2)
        st.dataframe(simple_payment, use_container_width=True)

    with col8:
        st.write("High-Risk Products:")
        simple_products = top_products[['transaction_id', 'fraud_rate']].round(2)
        simple_products.columns = ['Total Transactions', 'Fraud Rate %']
        st.dataframe(simple_products, use_container_width=True)

    # Simple Filters - REMOVED amount_category filter since it's commented out
    st.sidebar.header("🔍 Quick Filters")

    selected_payment = st.sidebar.selectbox(
        "View by Payment Method:",
        ['All'] + list(data['payment_method'].unique())
    )

    filtered_view = data.copy()
    if selected_payment != 'All':
        filtered_view = filtered_view[filtered_view['payment_method'] == selected_payment]

    st.sidebar.metric("Filtered Transactions", len(filtered_view))
    st.sidebar.metric("Filtered Fraud Rate", 
                      f"{(filtered_view['is_fraud'].sum() / len(filtered_view) * 100):.1f}%")

    # Key Insights in Clean Format
    st.subheader("💡 Key Insights")

    insight1, insight2, insight3 = st.columns(3)

    highest_fraud_payment = payment_summary.loc[payment_summary['fraud_rate'].idxmax()]
    highest_fraud_product = top_products.iloc[0]
    highest_fraud_location = top_locations.iloc[0]

    with insight1:
        st.info(f"Highest Fraud Payment: {highest_fraud_payment['payment_method']} ({highest_fraud_payment['fraud_rate']:.1f}%)")

    with insight2:
        st.warning(f"Riskiest Product: {highest_fraud_product.name} ({highest_fraud_product['fraud_rate']:.1f}%)")

    with insight3:
        st.error(f"Riskiest Location: {highest_fraud_location.name} ({highest_fraud_location['fraud_rate']:.1f}%)")

# --------------------------
# Enhanced Prediction Page
# --------------------------
def show_prediction_page(data):
    st.title("🔮 Enhanced Fraud Prediction")
    
    # Back to dashboard button - Centered and prominent
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Back to Dashboard", use_container_width=True, type="secondary"):
            st.session_state.show_prediction = False
            st.rerun()
    st.markdown("---")
    
    st.markdown("### Enter Transaction Details to Predict Fraud")
    st.info("🔍 *Enhanced detection now includes:* Amount patterns, Product value, and Risk combinations")
    
    # Train enhanced model
    model, label_encoders = train_enhanced_model(data)
    
    # Input form - ONLY the required features for prediction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.selectbox("Product Name*", options=data['product_name'].unique())
            # Changed to Indian Rupees with appropriate default value
            transaction_amount = st.number_input("Transaction Amount (₹)*", min_value=0.0, value=25000.0, step=1000.0)
            
        with col2:
            payment_method = st.selectbox("Payment Method*", options=data['payment_method'].unique())
            location = st.selectbox("Location*", options=data['location'].unique())
            
        # Optional fields (not used for prediction, just for display)
        st.markdown("---")
        st.markdown("#### Optional Information (For Reference Only)")
        col3, col4 = st.columns(2)
        with col3:
            transaction_id = st.number_input("Transaction ID", min_value=1, value=1)
            user_id = st.number_input("User ID", min_value=1, value=1000)
        with col4:
            transaction_date = st.date_input("Transaction Date")
            transaction_time = st.time_input("Transaction Time")
            
        st.markdown("*Required fields for prediction")
        
        # Submit button
        submitted = st.form_submit_button("Predict Fraud", use_container_width=True)
    
    # Prediction logic - This only runs when the form is submitted
    if submitted:
        # Prepare input data - ONLY the 4 required features
        input_data = {
            'transaction_amount': transaction_amount,
            'payment_method': payment_method,
            'location': location,
            'product_name': product_name
        }
        
        # Make enhanced prediction
        prediction, probability = predict_fraud_enhanced(model, label_encoders, input_data)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Enhanced Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.subheader("Transaction Details")
            st.write(f"*Product:* {product_name}")
            st.write(f"*Amount:* ₹{transaction_amount:,.2f}")
            st.write(f"*Payment Method:* {payment_method}")
            st.write(f"*Location:* {location}")
            
            # Display risk factors
            st.subheader("🔍 Risk Factors")
            risk_factors = []
            
            high_value_products = ['Smartphone', 'Laptop', 'Television', 'Camera', 'Gaming Console']
            high_risk_payments = ['Cash on Delivery']
            high_risk_locations = ['Delhi', 'Mumbai', 'Kolkata']
            
            if input_data['transaction_amount'] > 50000:
                risk_factors.append("💰 High transaction amount (> ₹50,000)")
            if product_name in high_value_products:
                risk_factors.append("📦 High-value product")
            if payment_method in high_risk_payments:
                risk_factors.append("🎯 High-risk payment method")
            if location in high_risk_locations:
                risk_factors.append("📍 High-risk location")
            if input_data['transaction_amount'] <= 10:
                risk_factors.append("⚠ Very low amount (suspicious)")
            if input_data['transaction_amount'] <= 1:
                risk_factors.append("🚨 Extremely low amount (highly suspicious)")
            
            for factor in risk_factors:
                st.write(f"• {factor}")
                
        with result_col2:
            st.subheader("Fraud Prediction")
            
            # Enhanced risk assessment
            risk_level = "LOW"
            if probability > 0.8:
                risk_level = "VERY HIGH"
            elif probability > 0.6:
                risk_level = "HIGH"
            elif probability > 0.4:
                risk_level = "MEDIUM"
            elif probability > 0.2:
                risk_level = "LOW"
            else:
                risk_level = "VERY LOW"
            
            if prediction == 1:
                st.error(f"🚨 *FRAUD DETECTED*")
                st.metric("Fraud Probability", f"{probability:.2%}", delta=f"Risk: {risk_level}")
                st.error("⚠ This transaction has been flagged as potentially fraudulent!")
                
                # Show override reason if applicable
                if hasattr(st.session_state, 'risk_override_reason') and st.session_state.risk_override_reason:
                    st.warning(f"**Override Applied:** {st.session_state.risk_override_reason}")
            else:
                st.success(f"✅ *LEGITIMATE TRANSACTION*")
                st.metric("Fraud Probability", f"{probability:.2%}", delta=f"Risk: {risk_level}")
                st.success("✓ This transaction appears to be legitimate.")
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 📊 Risk Analysis")
        
        # Compare with historical data for the 4 features
        product_risk = data[data['product_name'] == product_name]['is_fraud'].mean() * 100
        payment_risk = data[data['payment_method'] == payment_method]['is_fraud'].mean() * 100
        location_risk = data[data['location'] == location]['is_fraud'].mean() * 100
        amount_risk_data = data[np.abs(data['transaction_amount'] - transaction_amount) <= 5000]  # Increased tolerance for INR
        amount_risk = amount_risk_data['is_fraud'].mean() * 100 if len(amount_risk_data) > 0 else 0
        
        insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
        
        with insight_col1:
            st.metric("Product Risk", f"{product_risk:.1f}%")
            
        with insight_col2:
            st.metric("Payment Risk", f"{payment_risk:.1f}%")
            
        with insight_col3:
            st.metric("Location Risk", f"{location_risk:.1f}%")
            
        with insight_col4:
            st.metric("Amount Risk", f"{amount_risk:.1f}%")
    
    # # Enhanced test cases including high-risk combinations and ₹1 transactions
    # st.markdown("---")
    # st.markdown("### 🧪 High-Risk Test Cases")
    
    # high_risk_tests = [
    #     {"product": "Laptop", "amount": 85000.0, "payment": "Cash on Delivery", "location": "Delhi", "desc": "🚨 CRITICAL: Laptop + Cash + Delhi (HIGH RISK)"},
    #     {"product": "Smartphone", "amount": 1.0, "payment": "Cash on Delivery", "location": "Mumbai", "desc": "🚨 SUSPICIOUS: ₹1 Smartphone + Cash"},
    #     {"product": "Television", "amount": 65000.0, "payment": "Cash on Delivery", "location": "Kolkata", "desc": "🚨 HIGH RISK: TV + Cash + Kolkata"},
    #     {"product": "Camera", "amount": 45000.0, "payment": "Cash on Delivery", "location": "Delhi", "desc": "⚠ MEDIUM RISK: Camera + Cash + Delhi"},
    #     {"product": "Headphones", "amount": 3000.0, "payment": "UPI", "location": "Chennai", "desc": "✅ LOW RISK: Normal headphones purchase"},
    #     {"product": "Laptop", "amount": 1.0, "payment": "Credit Card", "location": "Bangalore", "desc": "🚨 SUSPICIOUS: ₹1 Laptop"}
    # ]
    
    # for i, test_case in enumerate(high_risk_tests):
    #     col1, col2 = st.columns([3, 1])
    #     with col1:
    #         st.write(f"**{test_case['desc']}**")
    #         st.write(f"Product: {test_case['product']} | Amount: ₹{test_case['amount']:,.2f} | Payment: {test_case['payment']} | Location: {test_case['location']}")
    #     with col2:
    #         if st.button(f"Test #{i+1}", key=f"hr_test_{i}"):
    #             st.session_state.test_sample = test_case
    #             st.rerun()
    
    # # Display current sample if loaded
    # if 'test_sample' in st.session_state:
    #     sample = st.session_state.test_sample
    #     st.info(f"📝 *Test case loaded*: {sample['desc']}")

# --------------------------
# Main App
# --------------------------
def main():
    # Initialize session state
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    
    # Load data
    data = load_data()
    
    # Show appropriate page based on session state
    if st.session_state.show_prediction:
        show_prediction_page(data)
    else:
        show_dashboard(data)

# Run the app
if __name__ == "__main__":
    main()