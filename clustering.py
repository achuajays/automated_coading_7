
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def preprocess_data(df, standardize=True):
    df = df.dropna()
    if standardize:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

def train_clustering(X, n_clusters=4, algorithm='K-means'):
    if algorithm == 'K-means':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Unsupported algorithm")
    
    labels = model.fit_predict(X)
    return model, labels

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette='viridis')
    plt.title('2D Cluster Visualization (PCA)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    return plt.gcf()
