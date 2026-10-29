# Lesson 12: Code Lab - Build a small ML project step by step
#
# Useful documentation:
# 1. pandas: https://pandas.pydata.org/docs/
# 2. train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 3. LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 4. accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 5. confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# 6. StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# 7. DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# 8. RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# Step 1: Build a small dataset
# --------------------------------------------------
# Students: Change this dataset or use your own project data
data = [
    {"hours_studied": 2, "previous_score": 50, "attendance": 60, "sleep_hours": 5, "passed": 0},
    {"hours_studied": 4, "previous_score": 65, "attendance": 75, "sleep_hours": 6, "passed": 1},
    {"hours_studied": 5, "previous_score": 70, "attendance": 80, "sleep_hours": 7, "passed": 1},
    {"hours_studied": 3, "previous_score": 55, "attendance": 65, "sleep_hours": 6, "passed": 0},
    {"hours_studied": 6, "previous_score": 80, "attendance": 90, "sleep_hours": 8, "passed": 1},
    {"hours_studied": 3, "previous_score": 60, "attendance": 70, "sleep_hours": 6, "passed": 0},
    {"hours_studied": 7, "previous_score": 84, "attendance": 88, "sleep_hours": 8, "passed": 1},
    {"hours_studied": 1, "previous_score": 45, "attendance": 55, "sleep_hours": 5, "passed": 0},
    {"hours_studied": 8, "previous_score": 90, "attendance": 95, "sleep_hours": 9, "passed": 1},
    {"hours_studied": 2, "previous_score": 52, "attendance": 65, "sleep_hours": 6, "passed": 0},
    {"hours_studied": 9, "previous_score": 92, "attendance": 96, "sleep_hours": 9, "passed": 1},
    {"hours_studied": 4, "previous_score": 61, "attendance": 72, "sleep_hours": 6, "passed": 0},
]

df = pd.DataFrame(data)

print("Step 1: Dataset created")
print(df)
print()

# --------------------------------------------------
# Step 2: Check and clean the data
# --------------------------------------------------
# Students: Use pandas docs to learn how to:
# - check missing values
# - fill missing values
# - remove useless columns
print("Step 2: Data cleaning")
print("Missing values:\n", df.isnull().sum())
print()

# Fill missing values if there were any
# (This dataset does not have any missing values, but students can add some)
df = df.fillna(df.median(numeric_only=True))

# --------------------------------------------------
# Step 3: Prepare features and target
# --------------------------------------------------
# Students: Read the documentation for pandas DataFrame and column selection
X = df[["hours_studied", "previous_score", "attendance", "sleep_hours"]]
y = df["passed"]

print("Step 3: Features and target prepared")
print("X columns:", X.columns.tolist())
print("Target values:", y.tolist())
print()

# --------------------------------------------------
# Step 4: Split into training and test data
# --------------------------------------------------
# Students: Read train_test_split docs to understand test_size and random_state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("Step 4: Train/test split complete")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print()

# --------------------------------------------------
# Step 5: Scale the data if needed
# --------------------------------------------------
# Students: Read the StandardScaler docs and explain why scaling may help
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Step 5: Data scaled")
print()

# --------------------------------------------------
# Step 6: Choose a model
# --------------------------------------------------
# Students: Find another model on scikit-learn and replace this one
model = LogisticRegression(max_iter=200)

print("Step 6: Model chosen")
print("Model:", type(model).__name__)
print()

# --------------------------------------------------
# Step 7: Train the model
# --------------------------------------------------
model.fit(X_train_scaled, y_train)

print("Step 7: Model trained")
print()

# --------------------------------------------------
# Step 8: Make predictions
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("Step 8: Predictions made")
print("Actual:", y_test.tolist())
print("Predicted:", y_pred.tolist())
print()

# --------------------------------------------------
# Step 9: Evaluate the model
# --------------------------------------------------
# Students: Read the accuracy_score docs and explain what it means
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Step 9: Model evaluation")
print("Accuracy:", round(accuracy, 3))
print("Confusion matrix:\n", cm)
print()

# --------------------------------------------------
# Step 10: Improve the model
# --------------------------------------------------
# Students: Try changing:
# - the model
# - the features
# - the data
# - the test_size
# - the random_state
#
# Example: try DecisionTreeClassifier
# Another model can be used to see if the results improve
better_model = DecisionTreeClassifier(max_depth=3, random_state=42)
better_model.fit(X_train_scaled, y_train)
y_pred_2 = better_model.predict(X_test_scaled)
accuracy_2 = accuracy_score(y_test, y_pred_2)

print("Step 10: Try another model")
print("DecisionTree accuracy:", round(accuracy_2, 3))
print()

# --------------------------------------------------
# Step 11: Make a prediction for one new student
# --------------------------------------------------
new_student = pd.DataFrame({
    "hours_studied": [6],
    "previous_score": [75],
    "attendance": [85],
    "sleep_hours": [7]
})

new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)[0]
result = "PASS" if prediction == 1 else "FAIL"

print("Step 11: New prediction")
print("New student data:", new_student.to_dict(orient="records"))
print("Prediction:", result)
print()

# --------------------------------------------------
# Step 12: Reflection
# --------------------------------------------------
# Students answer:
# 1. What did the model learn?
# 2. Why is accuracy useful?
# 3. What does the confusion matrix show?
# 4. Which model worked best?
# 5. What would you change to improve the project?