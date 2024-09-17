import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Customer Data.csv")

# Fill missing values with mean
df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean(), inplace=True)
df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean(), inplace=True)


df.drop(columns=["CUST_ID"], axis=1, inplace=True)


scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)


pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components, columns=["PCA1", "PCA2"])


kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_df)
clusters = kmeans.labels_

# Add cluster labels as a new column
df['Cluster'] = clusters

# Save the new supervised CSV file
df.to_csv("Supervised_Customer_Data.csv", index=False)

# Plotting (optional)
plt.figure(figsize=(8, 8))
sns.scatterplot(x="PCA1", y="PCA2", hue=clusters, palette="viridis", data=pca_df)
plt.title("Clusters visualization")
plt.show()
