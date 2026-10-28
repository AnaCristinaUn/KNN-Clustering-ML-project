import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from preprocessing import clean_transactions


data_path = Path(__file__).resolve().parents[1] / "data" / "chargebackFrauds.csv"
data = clean_transactions(pd.read_csv(data_path))

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
 
neighbors = np.array([1, 3, 5, 7, 9, 10])                                        # Array of different values for the number of neighbors to test in KNN.

for n_neighbors in neighbors:                                                     # Loop through each value of n_neighbors to train and evaluate the KNN model.
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))  # Creating a pipeline that first standardizes the features using StandardScaler and then applies KNeighborsClassifier with the current number of neighbors.
    model.fit(X_train, y_train)                                                   # Fitting the model to the training data, allowing it to learn the patterns in the features and their relationship to the target variable.

    val_predictions = model.predict(X_val)                                        # Making predictions on the validation set using the trained model to evaluate its performance before testing on the final test set.
    print(f"Validation results for {n_neighbors} neighbors:")
    print(classification_report(y_val, val_predictions, zero_division=0))

test_predictions = model.predict(X_test)                                          # Making predictions on the test set using the trained model to evaluate its performance on unseen data, providing an unbiased assessment of its generalization ability.
print("Test results:")
print(classification_report(y_test, test_predictions, zero_division=0))