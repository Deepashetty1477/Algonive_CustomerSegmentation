# ==========================================================
# Project: Customer Segmentation using Python
# Author: Deepa Shetty
# Internship: Algonive
# ==========================================================

# =============================
# Import Required Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# =============================
# Step 1: Folder Setup
# =============================
# Automatically create required folders
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =============================
# Step 2: Load Dataset
# =============================
# Try to find marketing_campaign.csv or customer_data.csv automatically
possible_files = [
    "data/raw/marketing_campaign.csv",
    "marketing_campaign.csv",
    "data/raw/customer_data.csv"
]

data_path = None
for path in possible_files:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("❌ Dataset not found! Please place your CSV file in:")
    print("   → data/raw/marketing_campaign.csv")
    exit()
else:
    print(f"✅ Dataset found at: {data_path}")

# Read data (Tab-separated format)
data = pd.read_csv(data_path, sep='\t')

print(f"✅ Data loaded successfully! Shape: {data.shape}")
print("\nColumns:\n", data.columns.tolist()[:10], "...")  # show first 10 column names
print("\nFirst 5 rows:\n", data.head())

# =============================
# Step 3: Data Cleaning
# =============================
print("\n🧹 Cleaning data...")

# Drop duplicates and missing values
data = data.drop_duplicates().dropna()

# Convert categorical variables to numeric (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)

print("✅ Data cleaned successfully!")
print(f"New shape after cleaning: {data.shape}")

# =============================
# Step 4: Feature Scaling
# =============================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("✅ Data scaled successfully!")

# =============================
# Step 5: Determine Optimal Clusters (Elbow Method)
# =============================
print("\n📈 Finding optimal number of clusters using Elbow Method...")
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.tight_layout()
plt.savefig("data/processed/elbow_method.png")
plt.show()  # 👈 show the Elbow Method plot
plt.close()

print("✅ Elbow Method plot saved at 'data/processed/elbow_method.png'")

# =============================
# Step 6: Apply K-Means Clustering
# =============================
optimal_k = 4  # You can adjust this based on the Elbow Method graph
print(f"\n🚀 Applying KMeans with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

print("✅ Clustering completed successfully!")
print("\nCluster Counts:\n", data['Cluster'].value_counts())

# =============================
# Step 7: Dimensionality Reduction (PCA)
# =============================
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
data['PCA1'] = pca_data[:, 0]
data['PCA2'] = pca_data[:, 1]

# =============================
# Step 8: Visualization
# =============================
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='Set2', s=60)
plt.title('Customer Segmentation Visualization (PCA)')

# Add cluster labels on plot
centers = kmeans.cluster_centers_
pca_centers = PCA(n_components=2).fit_transform(centers)
for i, (x, y) in enumerate(pca_centers):
    plt.text(x, y, f"Cluster {i}", fontsize=10, weight='bold', color='black', 
             bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

plt.savefig("data/processed/cluster_visualization.png")
plt.show()  # 👈 show the PCA cluster visualization
plt.close()

print("🎨 Cluster visualization saved at 'data/processed/cluster_visualization.png'")

# =============================
# Step 9: Save Clustered Data
# =============================
output_path = "data/processed/clustered_customers.csv"
data.to_csv(output_path, index=False)
print(f"✅ Clustered data saved successfully at '{output_path}'")

# =============================
# Step 10: Summary
# =============================
print("\n========= SUMMARY =========")
print("✅ Data loaded, cleaned, and scaled.")
print(f"✅ KMeans clustering applied with {optimal_k} clusters.")
print("✅ Visualization and results saved in data/processed/")
print("============================")
