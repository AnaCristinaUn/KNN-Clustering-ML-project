import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from preprocessing import add_transaction_velocity_features, clean_transactions


eps_values = [0.5, 0.8, 1.0]
min_samples_values = [5, 10]
plot_eps = 0.8
plot_min_samples = 5


data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"
data = clean_transactions(pd.read_csv(data_path))
data = add_transaction_velocity_features(data)

data["DayOfMonth"] = data["Date"].dt.day                                  # Extracting the day of the month from the "Date" column and creating a new column "DayOfMonth" in the DataFrame
data["Hour"] = data["Date"].dt.hour
velocity_features = data[
    [
        "CardTransactionCount",
        "TransactionsLastHour",
        "TransactionsLast24h",
        "TimeSincePreviousTransaction",
    ]
]

normal_features_with_amount = data[["Amount", "DayOfMonth", "Hour"]]
normal_features_without_amount = data[["DayOfMonth", "Hour"]]
hour_features_with_amount = data[["Amount", "Hour"]]
hour_features_without_amount = data[["Hour"]]

days_in_month = data["Date"].dt.days_in_month                             # Calculating the number of days in the month for each transaction to use in the cyclical transformation of the day and hour features.
day_angle = 2 * np.pi * (data["DayOfMonth"] - 1) / days_in_month          # Converting the day of the month to a cyclic angle.
hour_angle = 2 * np.pi * data["Hour"] / 24                                # Converting the hour of the day to a cyclic angle.

data["DaySin"] = np.sin(day_angle)                                        # Creating new columns "DaySin", "DayCos", "HourSin", and "HourCos" in the DataFrame to represent the cyclical transformations of the day and hour features using sine and cosine functions.
data["DayCos"] = np.cos(day_angle)
data["HourSin"] = np.sin(hour_angle)
data["HourCos"] = np.cos(hour_angle)

cyclical_features_with_amount = data[["Amount", "DaySin", "DayCos", "HourSin", "HourCos"]]
cyclical_features_without_amount = data[["DaySin", "DayCos", "HourSin", "HourCos"]]
cyclical_hour_features_with_amount = data[["Amount", "HourSin", "HourCos"]]
cyclical_hour_features_without_amount = data[["HourSin", "HourCos"]]
hour_features_with_velocity = hour_features_with_amount.join(velocity_features)
cyclical_hour_features_with_velocity = cyclical_hour_features_with_amount.join(
    velocity_features
)

feature_options = {
    "Normal date and hour with Amount": normal_features_with_amount,
    "Normal date and hour without Amount": normal_features_without_amount,
    "Normal hour without DayOfMonth with Amount": hour_features_with_amount,
    "Normal hour without DayOfMonth without Amount": hour_features_without_amount,
    "Cyclical date and hour with Amount": cyclical_features_with_amount,
    "Cyclical date and hour without Amount": cyclical_features_without_amount,
    "Cyclical hour without DayOfMonth with Amount": cyclical_hour_features_with_amount,
    "Cyclical hour without DayOfMonth without Amount": cyclical_hour_features_without_amount,
    "Normal hour without DayOfMonth with Amount and transaction velocity": hour_features_with_velocity,
    "Cyclical hour without DayOfMonth with Amount and transaction velocity": cyclical_hour_features_with_velocity,
}

scaler_options = {
    "StandardScaler": StandardScaler,                                          # Creating a dictionary "scaler_options" that maps scaler names to their corresponding scaler classes for later use in scaling the features before clustering.
    "RobustScaler": RobustScaler,
}

scaled_feature_options = {}                                                    # Creating an empty dictionary "scaled_feature_options" to store the scaled features for each combination of feature set and scaler.
for feature_name, features in feature_options.items():
    for scaler_name, scaler in scaler_options.items():
        option_name = f"{scaler_name} + {feature_name}"
        scaled_feature_options[option_name] = scaler().fit_transform(features)


def show_cluster_summary(data, labels, method, features):
    results = data.assign(
        Cluster=labels,
        IsChargeback=data["CBK"].eq("Yes"),
    )
    summary = results.groupby("Cluster").agg(                                  # Aggregating the results by cluster and calculating the number of transactions and median amount for each cluster
        Transactions=("Amount", "size"),
        Median_Amount=("Amount", "median"),
        CBK_Transactions=("IsChargeback", "sum"),
        CBK_Rate=("IsChargeback", "mean"),
    )

    print(f"\n{method} cluster summary:")
    print(summary.to_string(formatters={"CBK_Rate": "{:.1%}".format}))
    print("\nTransactions in each cluster:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    non_noise_clusters = summary.drop(index=-1, errors="ignore")
    if non_noise_clusters.empty:
        print("Chargeback classification metrics: not available (no non-noise clusters)")
    else:
        highest_rate_cluster = non_noise_clusters["CBK_Rate"].idxmax()
        predicted_chargebacks = labels == highest_rate_cluster
        actual_chargebacks = data["CBK"].eq("Yes").to_numpy()
        print(
            "Chargeback classification metrics "
            "(in-sample; highest-rate non-noise cluster):"
        )
        print(
            f"Accuracy:  {accuracy_score(actual_chargebacks, predicted_chargebacks):.4f}"
        )
        print(
            f"Precision: {precision_score(actual_chargebacks, predicted_chargebacks, zero_division=0):.4f}"
        )
        print(
            f"Recall:    {recall_score(actual_chargebacks, predicted_chargebacks, zero_division=0):.4f}"
        )
        print(
            f"F1 score:  {f1_score(actual_chargebacks, predicted_chargebacks, zero_division=0):.4f}"
        )

    clustered_points = labels != -1
    clustered_labels = labels[clustered_points]
    number_of_clusters = len(np.unique(clustered_labels))

    if 1 < number_of_clusters < len(clustered_labels):
        score = silhouette_score(
            features[clustered_points],
            clustered_labels,
            sample_size=min(2000, len(clustered_labels)),
            random_state=42,
        )
        print(f"Silhouette Score (sample, noise excluded): {score:.4f}")
    else:
        print("Silhouette Score: not available (need at least two clusters)")


def get_cluster_comparison_metrics(labels, features):
    clustered_points = labels != -1
    clustered_labels = labels[clustered_points]
    number_of_clusters = len(np.unique(clustered_labels))
    silhouette = None

    if 1 < number_of_clusters < len(clustered_labels):
        silhouette = silhouette_score(
            features[clustered_points],
            clustered_labels,
            sample_size=min(2000, len(clustered_labels)),
            random_state=42,
        )

    chargebacks = data["CBK"].eq("Yes").to_numpy()[clustered_points]
    cluster_summary = (
        pd.DataFrame(
            {
                "Cluster": clustered_labels,
                "IsChargeback": chargebacks,
            }
        )
        .groupby("Cluster")["IsChargeback"]
        .agg(["mean", "sum"])
    )
    highest_rate_cluster = cluster_summary["mean"].idxmax()

    return {
        "silhouette": silhouette,
        "chargeback_rate": cluster_summary.loc[highest_rate_cluster, "mean"],
        "chargeback_share": (
            cluster_summary.loc[highest_rate_cluster, "sum"]
            / data["CBK"].eq("Yes").sum()
        ),
    }


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

    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Day of month")
    axis.set_ylabel("Hour")
    axis.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)

kmeans_labels = {}
dbscan_labels = {}

for option_name, scaled_features in scaled_feature_options.items():
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # K-Means assigns every transaction to one of three clusters.
    kmeans_labels[option_name] = kmeans.fit_predict(scaled_features)
    show_cluster_summary(
        data,
        kmeans_labels[option_name],
        f"K-Means ({option_name})",
        scaled_features,
    )

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)   # DBSCAN groups nearby transactions and labels outliers as -1 (noise).
            labels = dbscan.fit_predict(scaled_features)

            print(f"\nDBSCAN group sizes ({option_name}, eps={eps}, min_samples={min_samples}):")
            print(pd.Series(labels).value_counts().sort_index().to_string())

            if eps == plot_eps and min_samples == plot_min_samples:
                dbscan_labels[option_name] = labels

    show_cluster_summary(
        data,
        dbscan_labels[option_name],
        f"DBSCAN ({option_name}, eps={plot_eps}, min_samples={plot_min_samples})",
        scaled_features,
    )

chronological_data = data.sort_values("Date", kind="stable")
split_position = int(len(chronological_data) * 0.8)
test_start = chronological_data["Date"].iloc[split_position]
train_mask = data["Date"] < test_start
test_mask = data["Date"] >= test_start
train_chargebacks = data.loc[train_mask, "CBK"].eq("Yes").to_numpy()
test_chargebacks = data.loc[test_mask, "CBK"].eq("Yes").to_numpy()

print("\nChronological K-Means holdout evaluation:")
print(
    f"Training: {int(train_mask.sum())} transactions before {test_start}; "
    f"test: {int(test_mask.sum())} transactions from {test_start} onward"
)

for feature_name, features in feature_options.items():
    for scaler_name, scaler_type in scaler_options.items():
        scaler = scaler_type()
        train_features = scaler.fit_transform(features.loc[train_mask])
        test_features = scaler.transform(features.loc[test_mask])

        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        train_labels = model.fit_predict(train_features)
        test_labels = model.predict(test_features)

        training_cluster_rates = (
            pd.DataFrame(
                {
                    "Cluster": train_labels,
                    "IsChargeback": train_chargebacks,
                }
            )
            .groupby("Cluster")["IsChargeback"]
            .mean()
        )
        highest_risk_cluster = training_cluster_rates.idxmax()
        predicted_chargebacks = test_labels == highest_risk_cluster
        test_cluster_chargeback_rate = (
            test_chargebacks[predicted_chargebacks].mean()
            if predicted_chargebacks.any()
            else 0.0
        )
        test_chargeback_capture = (
            (test_chargebacks & predicted_chargebacks).sum()
            / test_chargebacks.sum()
        )
        test_silhouette = None
        if 1 < len(np.unique(test_labels)) < len(test_labels):
            test_silhouette = silhouette_score(
                test_features,
                test_labels,
                sample_size=min(2000, len(test_labels)),
                random_state=42,
            )
        test_silhouette_label = (
            f"{test_silhouette:.4f}"
            if test_silhouette is not None
            else "not available"
        )

        print(
            f"\n{scaler_name} + {feature_name}: "
            f"selected training cluster {highest_risk_cluster} "
            f"(training CBK rate "
            f"{training_cluster_rates.loc[highest_risk_cluster]:.1%})"
        )
        print(
            f"Test metrics: "
            f"accuracy={accuracy_score(test_chargebacks, predicted_chargebacks):.4f}, "
            f"precision={precision_score(test_chargebacks, predicted_chargebacks, zero_division=0):.4f}, "
            f"recall={recall_score(test_chargebacks, predicted_chargebacks, zero_division=0):.4f}, "
            f"F1={f1_score(test_chargebacks, predicted_chargebacks, zero_division=0):.4f}"
        )
        print(
            f"Test selected-cluster CBK rate={test_cluster_chargeback_rate:.1%}; "
            f"CBKs captured={test_chargeback_capture:.1%}; "
            f"test silhouette={test_silhouette_label}"
        )

velocity_comparisons = {
    "Normal hour without DayOfMonth with Amount": (
        "Normal hour without DayOfMonth with Amount and transaction velocity"
    ),
    "Cyclical hour without DayOfMonth with Amount": (
        "Cyclical hour without DayOfMonth with Amount and transaction velocity"
    ),
}

print("\nBaseline vs transaction-velocity clustering comparison:")
for method, labels_by_option in [
    ("K-Means", kmeans_labels),
    ("DBSCAN", dbscan_labels),
]:
    print(f"\n{method}:")
    for scaler_name in scaler_options:
        for baseline_name, velocity_name in velocity_comparisons.items():
            baseline_option = f"{scaler_name} + {baseline_name}"
            velocity_option = f"{scaler_name} + {velocity_name}"
            baseline_metrics = get_cluster_comparison_metrics(
                labels_by_option[baseline_option],
                scaled_feature_options[baseline_option],
            )
            velocity_metrics = get_cluster_comparison_metrics(
                labels_by_option[velocity_option],
                scaled_feature_options[velocity_option],
            )
            print(
                f"{scaler_name}, {baseline_name}: "
                f"silhouette {baseline_metrics['silhouette']} -> "
                f"{velocity_metrics['silhouette']}; "
                f"highest-cluster CBK rate "
                f"{baseline_metrics['chargeback_rate']:.1%} -> "
                f"{velocity_metrics['chargeback_rate']:.1%}; "
                f"CBKs captured in that cluster "
                f"{baseline_metrics['chargeback_share']:.1%} -> "
                f"{velocity_metrics['chargeback_share']:.1%}"
            )

# Visualize the clusters
#
# plot_groups = {
#     "With Amount and DayOfMonth": [
#         "Normal date and hour with Amount",
#         "Cyclical date and hour with Amount",
#     ],
#     "Without Amount and with DayOfMonth": [
#         "Normal date and hour without Amount",
#         "Cyclical date and hour without Amount",
#     ],
#     "With Amount and without DayOfMonth": [
#         "Normal hour without DayOfMonth with Amount",
#         "Cyclical hour without DayOfMonth with Amount",
#     ],
#     "Without Amount and DayOfMonth": [
#         "Normal hour without DayOfMonth without Amount",
#         "Cyclical hour without DayOfMonth without Amount",
#     ],
# }
#
# for method, labels_by_option in [
#     ("K-Means", kmeans_labels),
#     ("DBSCAN", dbscan_labels),
# ]:
#     for group_name, feature_names in plot_groups.items():
#         figure, axes = plt.subplots(2, 2, figsize=(16, 12))
#         figure.suptitle(f"{method}: {group_name}", fontsize=14)
#         plot_number = 0
#
#         for feature_name in feature_names:
#             for scaler_name in scaler_options:
#                 option_name = f"{scaler_name} + {feature_name}"
#                 plot_title = f"{scaler_name}\n{feature_name}"
#                 show_clusters(
#                     axes.flat[plot_number],
#                     labels_by_option[option_name],
#                     plot_title,
#                 )
#                 plot_number += 1
#
#         figure.subplots_adjust(top=0.9, hspace=0.55, wspace=0.35)
#
# plt.show()
