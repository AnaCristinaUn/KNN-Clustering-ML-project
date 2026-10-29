import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from preprocessing import clean_transactions


data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"
data = clean_transactions(pd.read_csv(data_path))

data["DayOfMonth"] = data["Date"].dt.day                    # Extracting the day of the month from the "Date" column and creating a new column "DayOfMonth" in the DataFrame
data["Hour"] = data["Date"].dt.hour
features = data[["Amount", "DayOfMonth", "Hour"]]
scaled_features = StandardScaler().fit_transform(features)  # Scaling prevents the amount feature’s larger numbers from overpowering the day and hour features when the algorithms measure similarity.


def show_cluster_summary(data, labels, method):
    results = data.assign(Cluster=labels)
    summary = results.groupby("Cluster").agg(               # Aggregating the results by cluster and calculating the number of transactions and median amount for each cluster
        Transactions=("Amount", "size"),
        Median_Amount=("Amount", "median"),
    )

    print(f"\n{method} cluster summary:")
    print(summary.to_string())


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # K-Means assigns every transaction to one of three clusters.
kmeans_labels = kmeans.fit_predict(scaled_features)
show_cluster_summary(data, kmeans_labels, "K-Means")

dbscan = DBSCAN(eps=0.8, min_samples=5)                   # DBSCAN groups nearby transactions and labels outliers as -1 (noise).
dbscan_labels = dbscan.fit_predict(scaled_features)
show_cluster_summary(data, dbscan_labels, "DBSCAN")