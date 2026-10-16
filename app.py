import streamlit as st
import pandas as pd
import joblib

# using vs code:
# keep all the files in the same folder
# open the folder in vs code
# in a new terminal
# run app : .venv\Scripts\python.exe -m streamlit run app.py
# press enter again

@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler_model.pkl')
    return kmeans, scaler


kmeans, scaler = load_models()

FEATURES = ['Age', 'Income', 'Total_Spending', 'NumStorePurchases',
            'NumWebPurchases', 'NumWebVisitsMonth', 'Recency']

# Cluster numbers refer to the model saved by the notebook (random_state=42).
# If the model is retrained, the numbers should be checked again and  this dictionary should be updated.
CLUSTERS = {
    0: {
        "name": "Low value, recently active",
        "description": "They usually present as: "
                       "Mid-Low income and spending, few purchases, high web visits, "
                       "Fairly recent last purchase.",
        "action": "Engagement and increasing basket size.",
    },
    1: {
        "name": "Top spenders, mature and store-oriented",
        "description": "They usually present as: "
                       "Oldest segment, high income and very high spending, "
                       "mostly in-store purchases, few web visits, medium recency",
        "action": "Proactive contact and monitoring of the effects.",
    },
    2: {
        "name": "Mature, low-to-mid spenders",
        "description": "They usually present as: "
                       "middle aged or older customers, mid-low income, low spending, "
                       "few purchases, slightly more in-store than online.",
        "action": "Lower priority.",
    },
    3: {
        "name": "High value, omnichannel and digital",
        "description": "They usually present as: "
                       "High income and spending, store and web purchases at the same level, "
                       "high web visits, medium recency.",
        "action": "Monitor recency and keep them engaged.",
    },
    4: {
        "name": "Low value, at risk of churn",
        "description": "They usually present as: "
                       "mid-Low income and spending, few purchases, the highest web visits "
                       "and the highest recency. Possible price sensitivity (hypothesis only).",
        "action": "Reactivation and online promotions.",
    },
    5: {
        "name": "Top spenders, younger and store-oriented",
        "description": "They usually present as: "
                       "Highest income and spending, high store purchases, few web visits, "
                       "mid-high recency. Similar to cluster 1 but younger.",
        "action": "Proactive contact and a strategy tailored to a younger high-value customer.",
    },
}

st.title("Customer Segmentation App")
st.write("Enter data to predict the customer segment based on input features.")

Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Income = st.number_input("Income", min_value=0, value=50000)
Total_Spending = st.number_input("Total Spending (Sum of purchases)", min_value=0, value=1000)
Num_Web_Purchases = st.number_input("Number of Web Purchases", min_value=0, value=5)
Num_Store_Purchases = st.number_input("Number of Store Purchases", min_value=0, value=3)
Num_Web_Visits = st.number_input("Number of Web Visits (per month)", min_value=0, value=10)
Recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)

if st.button("Predict Customer Segment"):
    input_data = pd.DataFrame(
        [[Age, Income, Total_Spending, Num_Store_Purchases,
          Num_Web_Purchases, Num_Web_Visits, Recency]],
        columns=FEATURES,
    )
    input_data_scaled = scaler.transform(input_data)
    cluster = int(kmeans.predict(input_data_scaled)[0])

    info = CLUSTERS.get(cluster)
    if info:
        st.success(f"Predicted segment: Cluster {cluster} - {info['name']}")
        st.write(info["description"])
        st.info(f"Suggested action: {info['action']}")
    else:
        st.success(f"Predicted segment: Cluster {cluster}")
        st.warning("No description available for this cluster.")