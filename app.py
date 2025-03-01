
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
import streamlit as st
from config import title, background_color, text_color

st.set_page_config(page_title=title, page_icon="🌍", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title(title)

# Sidebar for configuration
st.sidebar.title("Configuration")
config_title = st.sidebar.text_input("App Title", value=title)
config_bg_color = st.sidebar.color_picker("Background Color", value=background_color)
config_text_color = st.sidebar.color_picker("Text Color", value=text_color)

# Update the config file if there are changes
if st.sidebar.button("Save Config"):
    with open('config.py', 'w') as f:
        f.write(f"title = '{config_title}'
")
        f.write(f"background_color = '{config_bg_color}'
")
        f.write(f"text_color = '{config_text_color}'
")
    st.success("Config saved. Please restart the app to see changes.")

st.write('This is a configurable Streamlit App.')
