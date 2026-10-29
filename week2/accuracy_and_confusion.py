# Lesson 11: Accuracy and Confusion Matrix
# You can change the dataset below to see:
# - good accuracy
# - bad accuracy
#
# Read these pages:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/model_evaluation.html

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------
# 1. Choose a dataset mode
# --------------------------------------------------
# Change this value to "good" or "bad"
mode = "good"

# --------------------------------------------------
# 2. Build two versions of the dataset
# --------------------------------------------------
# good_data: clear pattern, high accuracy
# bad_data: noisy and conflicting labels, lower accuracy
if mode == "good":
    data = [
        {"hours_studied": 1, "sleep_hours": 5, "passed": 0},
        {"hours_studied": 2, "sleep_hours": 6, "passed": 0},
        {"hours_studied": 3, "sleep_hours": 6, "passed": 0},
        {"hours_studied": 4, "sleep_hours": 7, "passed": 1},
        {"hours_studied": 5, "sleep_hours": 7, "passed": 1},
        {"hours_studied": 6, "sleep_hours": 8, "passed": 1},
        {"hours_studied": 7, "sleep_hours": 8, "passed": 1},
        {"hours_studied": 8, "sleep_hours": 9, "passed": 1},
        {"hours_studied": 9, "sleep_hours": 9, "passed": 1},
        {"hours_studied": 10, "sleep_hours": 8, "passed": 1},
    ]
else:
    data = [
        {"hours_studied": 1, "sleep_hours": 5, "passed": 0},
        {"hours_studied": 2, "sleep_hours": 6, "passed": 1},  # inconsistent
        {"hours_studied": 3, "sleep_hours": 6, "passed": 0},
        {"hours_studied": 4, "sleep_hours": 7, "passed": 0},  # inconsistent
        {"hours_studied": 5, "sleep_hours": 7, "passed": 1},
        {"hours_studied": 6, "sleep_hours": 8, "passed": 0},  # inconsistent
        {"hours_studied": 7, "sleep_hours": 8, "passed": 1},
        {"hours_studied": 8, "sleep_hours": 9, "passed": 0},  # inconsistent
        {"hours_studied": 9, "sleep_hours": 9, "passed": 1},
        {"hours_studied": 10, "sleep_hours": 8, "passed": 0},  # inconsistent
    ]

df = pd.DataFrame(data)

print("Step 2: Dataset chosen:", mode)
print(df)
print()

# --------------------------------------------------
# 3. Prepare X and y
# --------------------------------------------------
X = df[["hours_studied", "sleep_hours"]]
y = df["passed"]

# --------------------------------------------------
# 4. Split into training and testing data
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("Step 4: Data split complete")
print("Training size:", len(X_train))
print("Test size:", len(X_test))
print()

# --------------------------------------------------
# 5. Train a model
# --------------------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------
y_pred = model.predict(X_test)

print("Step 6: Predictions on test data")
print("Actual:", list(y_test))
print("Predicted:", list(y_pred))
print()

# --------------------------------------------------
# 7. Calculate accuracy
# --------------------------------------------------
# Students: Look up accuracy_score in the scikit-learn docs
accuracy = accuracy_score(y_test, y_pred)

print("Step 7: Accuracy")
print("Accuracy:", round(accuracy, 3))
print()

# --------------------------------------------------
# 8. Show confusion matrix
# --------------------------------------------------
# Students: Look up confusion_matrix in the docs
cm = confusion_matrix(y_test, y_pred)

print("Step 8: Confusion Matrix")
print(cm)

# Explain the matrix:
# rows = actual values
# columns = predicted values
#
# Example:
# [[TN, FP],
#  [FN, TP]]
#
# TN = true negative
# FP = false positive
# FN = false negative
# TP = true positive

# --------------------------------------------------
# 9. Make one new prediction
# --------------------------------------------------
new_student = [[5, 7]]
prediction = model.predict(new_student)[0]
label = "PASS" if prediction == 1 else "FAIL"

print()
print("Step 9: Predict one new student")
print("Features:", new_student)
print("Prediction:", label)