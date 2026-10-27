#importing necessary libraries
import pandas as pd
from pathlib import Path
from collections import Counter
# from preprocessing import preprocess_text

# Getting the dataset
data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"         # Relative path to the CSV file
data = pd.read_csv(data_path)                                                             # Reading the CSV file into a DataFrame

#Printing the number of samples in the dataset
print("Number of samples:", len(data))

# Showing the class distribution 
print("\nClass distribution:")
#value_counts() method counts the occurrences of each unique values in each column.
print(data["CBK"].value_counts())
print(data["Amount"].value_counts())
print(data["Date"].value_counts())
print(data["Card Number"].value_counts())

# Analyzing the "Amount" column
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

# Analyzing the Chargeback 
for (CBK, count) in data["CBK"].value_counts().items():                                    # Iterating through the unique values and their counts in the "CBK" column
    cbk_amounts = amounts[data["CBK"] == CBK]                                              # Filtering the "Amount" column based on the current "CBK" value
    median_cbk_amount = cbk_amounts.median()                                               # Calculating the median amount for the current "CBK" value
    percentage = count / len(data) * 100
    print(f"\nChargeback: {CBK} ({count} transactions, {percentage:.1f}%)")                # Printing the category, count, and share of all transactions
    print(f"Median amount: {median_cbk_amount:.2f}")                                       # Printing the median amount for the current "CBK" value