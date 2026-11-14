import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

df = pd.read_csv("sales_data_sample.csv", encoding = "latin1")
df = df.drop_duplicates()

features = ["QUANTITYORDERED", "PRICEEACH", "SALES"]
x = df[features].dropna()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


inertia = []

for k in range(1,11):
    model = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    model.fit(x_scaled)
    inertia.append(model.inertia_)

plt.plot(range(1,11), inertia, marker = "o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()


k = int(input("Enter optimal no of clusters: "))

kmeans = KMeans(n_clusters=k, random_state =42,n_init = 10)
labels = kmeans.fit_predict(x_scaled)

x_result = x.copy()
x_result["Cluster"] = labels

print("K-Means Cluster Count")
print(x_result["Cluster"].value_counts().sort_index())

print("Sample clustered data")
print(x_result.head(10))

hierarchical = AgglomerativeClustering(n_clusters = k)
x_result["Hierarchical_Cluster"] = hierarchical.fit_predict(x_scaled)

print("Hierarchical Cluster Counts: ")
print(x_result["Hierarchical_Cluster"].value_counts().sort_index())
