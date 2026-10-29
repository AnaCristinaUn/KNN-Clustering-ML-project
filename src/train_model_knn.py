import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from prepare_data import X_test, X_train, X_val, y_test, y_train, y_val

neighbors = [1, 2, 3, 4, 5, 7, 9, 10, 20, 30]

f1_scores_val = []  
accuracies_val = []
f1_scores_train = []
accuracies_train = []
f1_scores_test = []
accuracies_test = []                                                 

for n_neighbors in neighbors:                                                     # Loop through each value of n_neighbors to train and evaluate the KNN model.
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))  # Creating a pipeline that first standardizes the features using StandardScaler and then applies KNeighborsClassifier with the current number of neighbors.
    model.fit(X_train, y_train)                                                   # Fitting the model to the training data, allowing it to learn the patterns in the features and their relationship to the target variable.

    val_predictions = model.predict(X_val)                                        # Making predictions on the validation set using the trained model to evaluate its performance before testing on the final test set.
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    f1_scores_train.append(
        f1_score(y_train, train_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_train.append(accuracy_score(y_train, train_predictions))
    f1_scores_val.append(
        f1_score(y_val, val_predictions, pos_label="Yes", zero_division=0)        # Calculating the F1 score for the validation set predictions, specifically for the positive class "Yes", and appending it to the f1_scores_val list. The zero_division=0 parameter ensures that if there are no positive predictions, the F1 score will be set to 0 instead of raising an error.
    )
    accuracies_val.append(accuracy_score(y_val, val_predictions))

    f1_scores_test.append(
        f1_score(y_test, test_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_test.append(accuracy_score(y_test, test_predictions))
   
   
    # print(f"Validation results for {n_neighbors} neighbors:")
    # print(classification_report(y_val, val_predictions, zero_division=0))
    # test_predictions = model.predict(X_test)                                          # Making predictions on the test set using the trained model to evaluate its performance on unseen data, providing an unbiased assessment of its generalization ability.
    # print("Test results:")
    # print(classification_report(y_test, test_predictions, zero_division=0))

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

no_rows = y_train[y_train == "No"].sample(                                              # Randomly undersampling the majority class "No" to balance the dataset for training the KNN model. The number of samples taken from the "No" class is equal to the number of samples in the minority class "Yes"
    n=int(y_train.eq("Yes").sum()), random_state=42
)
undersampled_indices = y_train[y_train == "Yes"].index.union(no_rows.index)             # Combining the indices of the minority class "Yes" and the randomly sampled majority class "No" to create a new set of indices for the undersampled training dataset.
X_train_undersampled = X_train.loc[undersampled_indices]                                # Creating the undersampled training feature set by selecting only the rows corresponding to the combined indices of the minority and sampled majority classes.
y_train_undersampled = y_train.loc[undersampled_indices]

f1_scores_val_undersampled = []
accuracies_val_undersampled = []
f1_scores_train_undersampled = []
accuracies_train_undersampled = []
f1_scores_test_undersampled = []
accuracies_test_undersampled = []

for n_neighbors in neighbors:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))
    model.fit(X_train_undersampled, y_train_undersampled)

    train_predictions = model.predict(X_train_undersampled)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    f1_scores_train_undersampled.append(
        f1_score(y_train_undersampled, train_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_train_undersampled.append(
        accuracy_score(y_train_undersampled, train_predictions)
    )
    f1_scores_val_undersampled.append(
        f1_score(y_val, val_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_val_undersampled.append(accuracy_score(y_val, val_predictions))
    f1_scores_test_undersampled.append(
        f1_score(y_test, test_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_test_undersampled.append(accuracy_score(y_test, test_predictions))

plt.figure(figsize=(10, 6))

plt.plot(neighbors, f1_scores_val_undersampled, marker="o", label="Validation F1 (Yes)")
plt.plot(neighbors, f1_scores_test_undersampled, marker="o", label="Test F1 (Yes)")
plt.plot(neighbors, accuracies_val_undersampled, marker="s", label="Validation Accuracy")
plt.plot(neighbors, accuracies_test_undersampled, marker="s", label="Test Accuracy")

plt.xlabel("Number of neighbours (k)")
plt.ylabel("Score")
plt.title("KNN Undersampled Training Performance")
plt.xticks(neighbors)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

no_rows_3x = y_train[y_train == "No"].sample(
    n=3 * int(y_train.eq("Yes").sum()), random_state=42
)
undersampled_indices_3x = y_train[y_train == "Yes"].index.union(no_rows_3x.index)
X_train_undersampled_3x = X_train.loc[undersampled_indices_3x]
y_train_undersampled_3x = y_train.loc[undersampled_indices_3x]

f1_scores_val_undersampled_3x = []
accuracies_val_undersampled_3x = []
f1_scores_train_undersampled_3x = []
accuracies_train_undersampled_3x = []
f1_scores_test_undersampled_3x = []
accuracies_test_undersampled_3x = []

for n_neighbors in neighbors:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))
    model.fit(X_train_undersampled_3x, y_train_undersampled_3x)

    train_predictions = model.predict(X_train_undersampled_3x)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    f1_scores_train_undersampled_3x.append(
        f1_score(y_train_undersampled_3x, train_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_train_undersampled_3x.append(
        accuracy_score(y_train_undersampled_3x, train_predictions)
    )
    f1_scores_val_undersampled_3x.append(
        f1_score(y_val, val_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_val_undersampled_3x.append(accuracy_score(y_val, val_predictions))
    f1_scores_test_undersampled_3x.append(
        f1_score(y_test, test_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_test_undersampled_3x.append(accuracy_score(y_test, test_predictions))

plt.figure(figsize=(10, 6))

plt.plot(neighbors, f1_scores_val_undersampled_3x, marker="o", label="Validation F1 (Yes)")
plt.plot(neighbors, f1_scores_test_undersampled_3x, marker="o", label="Test F1 (Yes)")
plt.plot(neighbors, accuracies_val_undersampled_3x, marker="s", label="Validation Accuracy")
plt.plot(neighbors, accuracies_test_undersampled_3x, marker="s", label="Test Accuracy")

plt.xlabel("Number of neighbours (k)")
plt.ylabel("Score")
plt.title("KNN Training Performance (3 No per Yes)")
plt.xticks(neighbors)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

no_rows_half = y_train[y_train == "No"].sample(
    n=len(y_train[y_train == "No"]) // 2, random_state=42
)
undersampled_indices_half = y_train[y_train == "Yes"].index.union(no_rows_half.index)
X_train_undersampled_half = X_train.loc[undersampled_indices_half]
y_train_undersampled_half = y_train.loc[undersampled_indices_half]

f1_scores_val_undersampled_half = []
accuracies_val_undersampled_half = []
f1_scores_train_undersampled_half = []
accuracies_train_undersampled_half = []
f1_scores_test_undersampled_half = []
accuracies_test_undersampled_half = []

for n_neighbors in neighbors:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))
    model.fit(X_train_undersampled_half, y_train_undersampled_half)

    train_predictions = model.predict(X_train_undersampled_half)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    f1_scores_train_undersampled_half.append(
        f1_score(y_train_undersampled_half, train_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_train_undersampled_half.append(
        accuracy_score(y_train_undersampled_half, train_predictions)
    )
    f1_scores_val_undersampled_half.append(
        f1_score(y_val, val_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_val_undersampled_half.append(accuracy_score(y_val, val_predictions))
    f1_scores_test_undersampled_half.append(
        f1_score(y_test, test_predictions, pos_label="Yes", zero_division=0)
    )
    accuracies_test_undersampled_half.append(accuracy_score(y_test, test_predictions))

plt.figure(figsize=(10, 6))

plt.plot(neighbors, f1_scores_val_undersampled_half, marker="o", label="Validation F1 (Yes)")
plt.plot(neighbors, f1_scores_test_undersampled_half, marker="o", label="Test F1 (Yes)")
plt.plot(neighbors, accuracies_val_undersampled_half, marker="s", label="Validation Accuracy")
plt.plot(neighbors, accuracies_test_undersampled_half, marker="s", label="Test Accuracy")

plt.xlabel("Number of neighbours (k)")
plt.ylabel("Score")
plt.title("KNN Training Performance (Half of No Rows)")
plt.xticks(neighbors)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

results = {
    "Original training data": {
        "Training": {"Accuracy": accuracies_train, "F1": f1_scores_train},
        "Test": {"Accuracy": accuracies_test, "F1": f1_scores_test},
    },
    "Balanced training data": {
        "Training": {
            "Accuracy": accuracies_train_undersampled,
            "F1": f1_scores_train_undersampled,
        },
        "Test": {
            "Accuracy": accuracies_test_undersampled,
            "F1": f1_scores_test_undersampled,
        },
    },
    "3 No per Yes training data": {
        "Training": {
            "Accuracy": accuracies_train_undersampled_3x,
            "F1": f1_scores_train_undersampled_3x,
        },
        "Test": {
            "Accuracy": accuracies_test_undersampled_3x,
            "F1": f1_scores_test_undersampled_3x,
        },
    },
    "Half of No training data": {
        "Training": {
            "Accuracy": accuracies_train_undersampled_half,
            "F1": f1_scores_train_undersampled_half,
        },
        "Test": {
            "Accuracy": accuracies_test_undersampled_half,
            "F1": f1_scores_test_undersampled_half,
        },
    },
}

print("\nBest and worst scores by training strategy:")
for strategy, split_results in results.items():
    print(f"\n{strategy}:")
    for split_name, metrics in split_results.items():
        print(f"  {split_name}:")
        for metric_name, scores in metrics.items():
            best_index = int(np.argmax(scores))
            worst_index = int(np.argmin(scores))
            print(
                f"    {metric_name}: best = {scores[best_index]:.4f} "
                f"(k={neighbors[best_index]}), worst = {scores[worst_index]:.4f} "
                f"(k={neighbors[worst_index]})"
            )

plt.figure(figsize=(10, 6))

plt.plot(neighbors, f1_scores_test, marker="o", label="Original Test F1 (Yes)")
plt.plot(neighbors, f1_scores_test_undersampled, marker="o", label="Undersampled Test F1 (Yes)")
plt.plot(neighbors, f1_scores_test_undersampled_3x, marker="o", label="3 No per Yes Test F1 (Yes)")
plt.plot(neighbors, f1_scores_test_undersampled_half, marker="o", label="Half of No Rows Test F1 (Yes)")
plt.plot(neighbors, accuracies_test, marker="s", label="Original Test Accuracy")
plt.plot(neighbors, accuracies_test_undersampled, marker="s", label="Undersampled Test Accuracy")
plt.plot(neighbors, accuracies_test_undersampled_3x, marker="s", label="3 No per Yes Test Accuracy")
plt.plot(neighbors, accuracies_test_undersampled_half, marker="s", label="Half of No Rows Test Accuracy")

plt.xlabel("Number of neighbours (k)")
plt.ylabel("Score")
plt.title("KNN Test Performance Comparison")
plt.xticks(neighbors)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
