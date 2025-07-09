import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set the page title and layout
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🧠 Customer Segmentation App")
st.markdown("Choose how you'd like to input customer data:")

# Load scaler and model
@st.cache_resource
def load_scaler_model():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_scaler_model()

# Features expected by model
FEATURES = [
    'Age', 'Income', 'Total_Children', 'Total_Spend', 'Avg_Annual_Spend',
    'Recency', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
]

# Input mode
mode = st.radio("Select input mode", ["📁 Upload File", "🎛️ Manual Input"])

# -------------------- FILE UPLOAD MODE --------------------
if mode == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload your customer dataset", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("Raw Uploaded Data")
            st.dataframe(df.head(50))


            # Feature Engineering
            df['Age'] = 2025 - df['Year_Birth']
            df['Total_Children'] = df['Kidhome'] + df['Teenhome']

            spend_cols = [
                'MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
            ]   
            df['Total_Spend'] = df[spend_cols].sum(axis=1)

            df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
            df['Customer_Since_Days'] = (datetime.now() - df['Dt_Customer']).dt.days
            df['Avg_Annual_Spend'] = df['Total_Spend'] / (df['Customer_Since_Days'] / 365 + 1)

            # Select features for prediction
            X = df[FEATURES]

            # 🔧 Drop rows with any NaNs in feature columns
            X = X.dropna()

            # Scale and predict
            X_scaled = scaler.transform(X)
            clusters = model.predict(X_scaled)

            # Map back predicted clusters to the original dataframe
            df_filtered = df.loc[X.index].copy()
            df_filtered['Cluster'] = clusters


            st.success("✅ Segmentation complete!")
            st.subheader("Clustered Data")
            st.dataframe(df_filtered.head(50))

            # Visualization
            st.subheader("📊 Cluster Visualization")
            col1, col2 = st.columns(2)

            with col1:
                sns.countplot(x='Cluster', data=df_filtered)
                plt.title("Number of Customers per Cluster")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                sns.boxplot(x='Cluster', y='Income', data=df_filtered)
                plt.title("Income by Cluster")
                st.pyplot(plt.gcf())
                plt.clf()


            # Download segmented data
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Segmented Data", csv, file_name="segmented_customers.csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")






# -------------------- MANUAL INPUT MODE --------------------
elif mode == "🎛️ Manual Input":
    st.subheader("Enter Customer Info Manually")

    user_input = {
        'Age': st.slider("Age", 60, 100, 35),
        'Income': st.slider("Income", 0, 100000, 71613, step=100),
        'Total_Children': st.slider("Total Children", 0, 10, 2),
        'Total_Spend': st.slider("Total Spend", 0, 100000, 716, step=500),
        'Avg_Annual_Spend': st.slider("Avg Annual Spend", 0, 100000, 60, step=500),
        'Recency': st.slider("Recency (days since last purchase)", 0, 99, 26),
        'NumWebPurchases': st.slider("Web Purchases", 0, 50, 8),
        'NumCatalogPurchases': st.slider("Catalog Purchases", 0, 30, 2),
        'NumStorePurchases': st.slider("Store Purchases", 0, 30, 10),
    }

    if st.button("🔍 Predict Cluster"):
        try:
            user_df = pd.DataFrame([user_input])
            user_scaled = scaler.transform(user_df)
            cluster = model.predict(user_scaled)[0]
            st.success(f"🎯 This customer belongs to **Cluster {cluster}**.")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
