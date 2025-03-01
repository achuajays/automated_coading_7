
import streamlit as st
import pandas as pd

def render_ui(config):
    st.title(config.app_title)

    uploaded_file = st.file_uploader("Upload dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("Preview of the dataset:")
        st.dataframe(df.head())

        # Feature selection
        feature_columns = st.multiselect("Select features for clustering", df.columns)
        if feature_columns:
            # Number of clusters
            num_clusters = st.number_input(
                "Enter the number of clusters", min_value=2, max_value=10, value=config.default_clusters, step=1)

            # Perform clustering
            if st.button("Cluster"):
                return df, feature_columns, num_clusters
    return None, None, None
