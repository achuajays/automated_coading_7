
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from utils import load_data, perform_clustering, plot_clusters
from config import Config

# Load configuration settings
config = Config()

st.title("Customer Segmentation App")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.markdown("Preview of the dataset:")
    st.dataframe(df.head())

    # Feature selection
    feature_columns = st.multiselect("Select features for clustering", df.columns)
    if feature_columns:
        # Number of clusters
        num_clusters = st.number_input("Enter the number of clusters", min_value=2, max_value=10, value=3, step=1)

        # Perform clustering
        if st.button("Cluster"):
            kmeans, labels = perform_clustering(df[feature_columns], num_clusters)
            st.markdown("Cluster labels:")
            st.dataframe(labels)

            # Plot clusters
            plot_clusters(df, kmeans, labels, feature_columns)
