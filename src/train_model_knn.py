import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from prepare_data import X_test, X_train, X_val, y_test, y_train, y_val
 
neighbors = [1, 2, 3, 4, 5, 7, 9, 10, 20, 30]

f1_scores_val = []  
accuracies_val = []
f1_scores_test = []
accuracies_test = []                                                 

for n_neighbors in neighbors:                                                     # Loop through each value of n_neighbors to train and evaluate the KNN model.
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))  # Creating a pipeline that first standardizes the features using StandardScaler and then applies KNeighborsClassifier with the current number of neighbors.
    model.fit(X_train, y_train)                                                   # Fitting the model to the training data, allowing it to learn the patterns in the features and their relationship to the target variable.

    val_predictions = model.predict(X_val)                                        # Making predictions on the validation set using the trained model to evaluate its performance before testing on the final test set.
    test_predictions = model.predict(X_test)

    f1_scores_val.append(
        f1_score(y_val, val_predictions, pos_label="Yes", zero_division=0)        # Calculating the F1 score for the validation set predictions, specifically for the positive class "Yes", and appending it to the f1_scores_val list. The zero_division=0 parameter ensures that if there are no positive predictions, the F1 score will be set to 0 instead of raising an error.
    )
    accuracies_val.append(accuracy_score(y_val, val_predictions))

    f1_scores_test.append(
        f1_score(y_test, test_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_test.append(accuracy_score(y_test, test_predictions))
   
   
    print(f"Validation results for {n_neighbors} neighbors:")
    print(classification_report(y_val, val_predictions, zero_division=0))
    test_predictions = model.predict(X_test)                                          # Making predictions on the test set using the trained model to evaluate its performance on unseen data, providing an unbiased assessment of its generalization ability.
    print("Test results:")
    print(classification_report(y_test, test_predictions, zero_division=0))

plt.figure(figsize=(10, 6))

plt.plot(neighbors, f1_scores_val, marker="o", label="Validation F1 (Yes)")
plt.plot(neighbors, f1_scores_test, marker="o", label="Test F1 (Yes)")
plt.plot(neighbors, accuracies_val, marker="s", label="Validation Accuracy")
plt.plot(neighbors, accuracies_test, marker="s", label="Test Accuracy")

plt.xlabel("Number of neighbours (k)")
plt.ylabel("Score")
plt.title("KNN Validation vs Test Performance")
plt.xticks(neighbors)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
