#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
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

# Comparing amount distributions and chargeback rates
amount_groups = pd.qcut(amounts, q=4)                                                      # Dividing the "Amount" column into 4 quantile-based groups
# A quantile is a value that divides a dataset into groups containing approximately equal numbers of observations.
# In this case, qcut is used to create 4 groups (quartiles) based on the distribution of transaction amounts. 25%
chargeback_rates = data.groupby(amount_groups, observed=True)["CBK"].apply(                # Grouping the data by the amount groups and calculating the chargeback rate for each group
    lambda labels: labels.eq("Yes").mean()                                                 # Calculating the mean of the boolean series where "CBK" is "Yes" to get the chargeback rate
)

print("\nChargeback rate by amount group:")
for amount_group, rate in chargeback_rates.items():
    print(f"{amount_group}: {rate:.1%}")

data.boxplot(column="Amount", by="CBK")                                                    # Creating a boxplot of the "Amount" column grouped by the "CBK" column
plt.title("Transaction amounts by chargeback status")                                      # Setting the title of the boxplot
plt.suptitle("")                                                                           # Removing the default title to avoid overlap with the main title
plt.xlabel("Chargeback (CBK)")                                                             # Setting the x-axis label of the boxplot
plt.ylabel("Amount")                                                                       # Setting the y-axis label of the boxplot

# Analyzing daily transaction and chargeback patterns
dated_data = data.assign(Date=pd.to_datetime(data["Date"]))                                # Converting the "Date" column to datetime format and creating a new DataFrame with the converted dates
daily = dated_data.set_index("Date").resample("D").agg(                                    # Resampling the data by day and aggregating the number of transactions and chargeback rate for each day
    Transactions=("CBK", "size"),
    Chargeback_Rate=("CBK", lambda labels: labels.eq("Yes").mean()),
)

print("\nDate summary:")
print(f"Date range: {dated_data['Date'].min()} to {dated_data['Date'].max()}")
print(f"Days with transactions: {len(daily)}")
print(f"Average transactions per day: {daily['Transactions'].mean():.1f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)                               # Creating a figure with two subplots (axes) arranged vertically, sharing the x-axis
daily["Transactions"].plot(ax=axes[0], marker="o")                                         # Plotting the daily number of transactions on the first subplot with markers for each data point
axes[0].set_title("Transactions per day")                                                  # Setting the title of the first subplot
axes[0].set_ylabel("Transactions")                                                         # Setting the y-axis label of the first subplot
daily["Chargeback_Rate"].plot(ax=axes[1], marker="o")                                      # Plotting the daily chargeback rate on the second subplot with markers for each data point
axes[1].set_title("Daily chargeback rate")
axes[1].set_ylabel("Rate")
axes[1].set_xlabel("Day of May 2015")
axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d"))
axes[1].tick_params(axis="x", labelrotation=90)
fig.tight_layout()

# Showing the distribution of transaction amounts
plt.figure()
plt.hist(amounts, bins=30, edgecolor="black")                                             # Creating a histogram of the transaction amounts with 30 bins and black edges for better visibility
plt.title("Distribution of transaction amounts")
plt.xlabel("Amount")
plt.ylabel("Number of transactions")
plt.tight_layout()

plt.show()