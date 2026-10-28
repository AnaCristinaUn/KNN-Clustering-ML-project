import pandas as pd


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