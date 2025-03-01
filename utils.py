
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(file):
    return pd.read_csv(file)

def perform_clustering(data, num_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    return kmeans, labels

def plot_clusters(df, kmeans, labels, feature_columns):
    df['Cluster'] = labels
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, vars=feature_columns, hue='Cluster', palette='viridis')
    plt.title("Cluster Visualization")
    plt.show()
    plt.close()
