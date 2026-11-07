import numpy as np
import pandas as pd


def add_transaction_velocity_features(data):
    """Add per-card activity features based only on earlier transactions."""
    if not {"Card Number", "Date"}.issubset(data.columns):
        raise ValueError("Data must contain 'Card Number' and 'Date' columns")
    if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        raise TypeError("'Date' must contain datetime values")

    feature_columns = [
        "CardTransactionCount",
        "TransactionsLastHour",
        "TransactionsLast24h",
        "TimeSincePreviousTransaction",
    ]
    original_index = data.index
    chronological_data = data.reset_index(drop=True).sort_values(
        ["Card Number", "Date"], kind="stable"
    )
    features = np.zeros((len(data), len(feature_columns)))

    card_groups = chronological_data.groupby("Card Number", sort=False)
    for positions in card_groups.indices.values():
        card_transactions = chronological_data.iloc[positions]
        timestamps = card_transactions["Date"].to_numpy(dtype="datetime64[ns]")
        previous = timestamps.searchsorted(timestamps, side="left")
        previous_hour = timestamps.searchsorted(
            timestamps - pd.Timedelta(hours=1).to_timedelta64(),
            side="left",
        )
        previous_day = timestamps.searchsorted(
            timestamps - pd.Timedelta(hours=24).to_timedelta64(),
            side="left",
        )
        previous_timestamps = timestamps[np.maximum(previous - 1, 0)]
        elapsed_seconds = (
            (timestamps - previous_timestamps) / pd.Timedelta(seconds=1)
        )
        elapsed_seconds[previous == 0] = 0

        features[positions] = np.column_stack(
            (
                previous,
                previous - previous_hour,
                previous - previous_day,
                elapsed_seconds,
            )
        )

    chronological_data[feature_columns] = features
    return chronological_data.sort_index().set_axis(original_index)


def clean_transactions(data):
    data = data.copy()
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")                        # Dropping the "Unnamed: 0" column 

    required_columns = ["Card Number", "Date", "Amount", "CBK"]
    missing_columns = set(required_columns) - set(data.columns)                      # Checking for missing required columns in the dataset
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    missing_rows = data[required_columns].isna().any(axis=1)                         # Identifying rows with missing values in the required columns

    data["Card Number"] = data["Card Number"].astype("string").str.strip()           # Stripping whitespace from the "Card Number" column and converting it to string type
    data["CBK"] = data["CBK"].astype("string").str.strip()                           # Stripping whitespace from the "CBK" column and converting it to string type
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")                  # Converting the "Amount" column to numeric values, coercing errors to NaN
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")                     # Converting the "Date" column to datetime values, coercing errors to NaT

    invalid_rows = (                                                                 # Identifying rows with invalid values in the required columns
        data["Amount"].isna()                                                        # Identifying rows where the "Amount" column is NaN (invalid)
        | data["Amount"].le(0)                                                       # Identifying rows where the "Amount" column is less than or equal to 0 (invalid)
        | data["Date"].isna()                                                        # Identifying rows where the "Date" column is NaT (invalid)
        | data["Card Number"].eq("")                                                 # Identifying rows where the "Card Number" column is empty (invalid)
        | ~data["CBK"].isin(["No", "Yes"])                                           # Identifying rows where the "CBK" column has invalid values (invalid)
    ) & ~missing_rows                                                                # Ensuring that rows with missing values are not counted as invalid, since they will be removed anyway
    rows_to_remove = missing_rows | invalid_rows                                    

    print("Initial rows:", len(data))
    print("Rows with missing values:", int(missing_rows.sum()))
    print("Rows with invalid values:", int(invalid_rows.sum()))
    print("Rows removed:", int(rows_to_remove.sum()))

    cleaned_data = data.loc[~rows_to_remove].copy()
    duplicate_count = cleaned_data.duplicated(subset=required_columns).sum()
    cleaned_data = cleaned_data.drop_duplicates(subset=required_columns).copy()
    print("Duplicate transaction rows removed:", int(duplicate_count))
    print("Final rows:", len(cleaned_data))

    return cleaned_data