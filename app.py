import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Page Config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features for Clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

st.subheader("Elbow Method")

fig1, ax1 = plt.subplots()

ax1.plot(range(1, 11), wcss, marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")

st.pyplot(fig1)

# KMeans
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# Visualization
st.subheader("Customer Segments")

fig2, ax2 = plt.subplots(figsize=(8,6))

ax2.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=y_kmeans
)

ax2.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    marker='X'
)

ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score")
ax2.set_title("Customer Segmentation")

st.pyplot(fig2)

# Cluster Information
df["Cluster"] = y_kmeans

st.subheader("Clustered Data")

st.dataframe(df.head(20))

st.subheader("Cluster Summary")

summary = df.groupby("Cluster").mean(
    numeric_only=True
)

st.dataframe(summary)