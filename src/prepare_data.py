import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocessing import clean_transactions


data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"
data = clean_transactions(pd.read_csv(data_path))

card_numbers = data["Card Number"]                                   # Storing the "Card Number" column separately for later use in calculating card frequency.
data["Date"] = pd.to_datetime(data["Date"])                          # Converting the "Date" column to datetime format 


X = pd.DataFrame({                                                   # Creating a new DataFrame X with selected features for the model
    "Amount": data["Amount"],
    "DayOfMonth": data["Date"].dt.day,
    "Hour": data["Date"].dt.hour,
})

y = data["CBK"]                                                       # CBK is what the model should predict.

X_train_val, X_test, y_train_val, y_test = train_test_split(         # First reserve 20% for the final test.
    X, y, test_size=0.2, random_state=42, stratify=y                 # random_state ensures reproducibility, stratify ensures the class distribution is preserved in the splits.
)

X_train, X_val, y_train, y_val = train_test_split(                   # Split the remaining 80% into 60% training and 20% validation.
    X_train_val, y_train_val, test_size=0.25,
    random_state=42, stratify=y_train_val
)

card_frequency = card_numbers.loc[X_train.index].value_counts()      # Calculating the frequency of each card number in the training set to use as a feature in the model. 

# Map each transaction's actual card number to its frequency.
X_train = X_train.copy()                                             # Creating copies of the training, validation, and test sets to avoid modifying the original data during feature engineering.
X_val = X_val.copy()
X_test = X_test.copy()

X_train["CardFrequency"] = (
    card_numbers.loc[X_train.index].map(card_frequency).astype(float)# Mapping the card numbers in the training set to their corresponding frequencies and adding this as a new feature "CardFrequency" to the training DataFrame.
)

X_val["CardFrequency"] = (
    card_numbers.loc[X_val.index].map(card_frequency).fillna(0).astype(float)
)

X_test["CardFrequency"] = (
    card_numbers.loc[X_test.index].map(card_frequency).fillna(0).astype(float)
)