#importing necessary libraries
import pandas as pd
from pathlib import Path
from collections import Counter
# from preprocessing import preprocess_text

data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"         # Relative path to the CSV file
data = pd.read_csv(data_path)                                                             # Reading the CSV file into a DataFrame

print("Number of samples:", len(data))

amounts = pd.to_numeric(data["Amount"], errors="raise")                                   # Converting the "Amount" column to numeric values, raising an error if conversion fails
median_amount = amounts.median()
lowest_amount = amounts.min()
highest_amount = amounts.max()

print("\nCharge amount summary:")
print(f"Median: {median_amount:.2f}")
print(f"Lowest: {lowest_amount:.2f} ({(amounts == lowest_amount).sum()} transactions)")   # Counting the number of transactions with the lowest amount
print(f"Highest: {highest_amount:.2f} ({(amounts == highest_amount).sum()} transactions)")# Counting the number of transactions with the highest amount
print(
    f"At or below median: {(amounts <= median_amount).sum()} transactions "               # Counting the number of transactions at or below the median amount
    f"({(amounts <= median_amount).mean():.1%})"                                          # Calculating the percentage of transactions at or below the median amount
)
print(
    f"Above median: {(amounts > median_amount).sum()} transactions "                      # Counting the number of transactions above the median amount
    f"({(amounts > median_amount).mean():.1%})"                                           # Calculating the percentage of transactions above the median amount
)

print("\nClass distribution:")
#value_counts() method counts the occurrences of each unique values in each column.
print(data["CBK"].value_counts())
print(data["Amount"].value_counts())
print(data["Date"].value_counts())
print(data["Card Number"].value_counts())
