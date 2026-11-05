import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from preprocessing import clean_transactions


data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"
data = clean_transactions(pd.read_csv(data_path))

data["DayOfMonth"] = data["Date"].dt.day                                  # Extracting the day of the month from the "Date" column and creating a new column "DayOfMonth" in the DataFrame
data["Hour"] = data["Date"].dt.hour

normal_features = data[["Amount", "DayOfMonth", "Hour"]]
scaled_normal_features = StandardScaler().fit_transform(normal_features)  # Scaling prevents the amount feature’s larger numbers from overpowering the day and hour features when the algorithms measure similarity.

days_in_month = data["Date"].dt.days_in_month                             # Calculating the number of days in the month for each transaction to use in the cyclical transformation of the day and hour features.
day_angle = 2 * np.pi * (data["DayOfMonth"] - 1) / days_in_month          # Converting the day of the month to a cyclic angle.
hour_angle = 2 * np.pi * data["Hour"] / 24                                # Converting the hour of the day to a cyclic angle.

data["DaySin"] = np.sin(day_angle)                                        # Creating new columns "DaySin", "DayCos", "HourSin", and "HourCos" in the DataFrame to represent the cyclical transformations of the day and hour features using sine and cosine functions.
data["DayCos"] = np.cos(day_angle)
data["HourSin"] = np.sin(hour_angle)
data["HourCos"] = np.cos(hour_angle)

cyclical_features = data[["Amount", "DaySin", "DayCos", "HourSin", "HourCos"]] # Creating a new DataFrame "cyclical_features" that contains the amount and the cyclical transformations of the day and hour features.
scaled_cyclical_features = StandardScaler().fit_transform(cyclical_features)


def show_cluster_summary(data, labels, method):
    results = data.assign(Cluster=labels)
    summary = results.groupby("Cluster").agg(                                  # Aggregating the results by cluster and calculating the number of transactions and median amount for each cluster
        Transactions=("Amount", "size"),
        Median_Amount=("Amount", "median"),
    )

    print(f"\n{method} cluster summary:")
    print(summary.to_string())
    print("\nTransactions in each cluster:")
    print(pd.Series(labels).value_counts().sort_index().to_string())


def show_clusters(axis, labels, title):
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("tab20", len(unique_labels))

    for color_number, label in enumerate(unique_labels):
        points_in_cluster = labels == label
        cluster_name = f"Cluster {label}"
        marker = "o"
        color = colors(color_number)

        if label == -1:
            cluster_name = "Noise (-1)"
            marker = "x"
            color = "black"

        axis.scatter(
            data.loc[points_in_cluster, "DayOfMonth"],
            data.loc[points_in_cluster, "Hour"],
            label=cluster_name,
            color=color,
            marker=marker,
            s=15,
        )

    axis.set_title(title)
    axis.set_xlabel("Day of month")
    axis.set_ylabel("Hour")
    axis.legend()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)                     # K-Means assigns every transaction to one of three clusters.
kmeans_normal_labels = kmeans.fit_predict(scaled_normal_features)
show_cluster_summary(data, kmeans_normal_labels, "K-Means (normal date and hour)")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_cyclical_labels = kmeans.fit_predict(scaled_cyclical_features)
show_cluster_summary(data, kmeans_cyclical_labels, "K-Means (cyclical date and hour)")

dbscan = DBSCAN(eps=0.8, min_samples=5)                                       # DBSCAN groups nearby transactions and labels outliers as -1 (noise).
dbscan_normal_labels = dbscan.fit_predict(scaled_normal_features)
show_cluster_summary(data, dbscan_normal_labels, "DBSCAN (normal date and hour)")

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_cyclical_labels = dbscan.fit_predict(scaled_cyclical_features)
show_cluster_summary(data, dbscan_cyclical_labels, "DBSCAN (cyclical date and hour)")

# Visualize the clusters

figure, axes = plt.subplots(2, 2, figsize=(12, 10))                           # Comparing normal and cyclical date and hour features for both clustering methods.

show_clusters(axes[0, 0], kmeans_normal_labels, "K-Means: normal date and hour")
show_clusters(axes[0, 1], kmeans_cyclical_labels, "K-Means: cyclical date and hour")
show_clusters(
    axes[1, 0],
    dbscan_normal_labels,
    "DBSCAN: normal date and hour",
)
show_clusters(
    axes[1, 1],
    dbscan_cyclical_labels,
    "DBSCAN: cyclical date and hour",
)

figure.tight_layout()
plt.show()
