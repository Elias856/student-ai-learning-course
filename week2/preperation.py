'''
fills missing values
converts text categories to numbers
prepares features for a model
trains a regression model to predict final exam score
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# 1. Create a small student dataset
# --------------------------------------------------
data = [
    {"hours_studied": 4, "previous_score": 65, "attendance": 80, "study_group": "A", "final_score": 70},
    {"hours_studied": 6, "previous_score": 75, "attendance": 85, "study_group": "B", "final_score": 80},
    {"hours_studied": 5, "previous_score": 68, "attendance": 78, "study_group": "A", "final_score": 72},
    {"hours_studied": None, "previous_score": 82, "attendance": 90, "study_group": "C", "final_score": 88},
    {"hours_studied": 8, "previous_score": 88, "attendance": 95, "study_group": "B", "final_score": 92},
    {"hours_studied": 3, "previous_score": 60, "attendance": 70, "study_group": "A", "final_score": 66},
    {"hours_studied": 7, "previous_score": 79, "attendance": None, "study_group": "C", "final_score": 85},
    {"hours_studied": 9, "previous_score": 91, "attendance": 96, "study_group": "B", "final_score": 95},
    {"hours_studied": 2, "previous_score": 55, "attendance": 65, "study_group": "A", "final_score": 60},
    {"hours_studied": 5, "previous_score": 70, "attendance": 75, "study_group": "B", "final_score": 74},
]

df = pd.DataFrame(data)

print("Original data:")
print(df)
print()

# --------------------------------------------------
# 2. Data cleaning
# --------------------------------------------------
# Fill missing numeric values with median or mean
df["hours_studied"] = df["hours_studied"].fillna(df["hours_studied"].median())
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())

# Convert text categories into numbers
study_group_map = {"A": 0, "B": 1, "C": 2}
df["study_group"] = df["study_group"].map(study_group_map)

# Optional: remove irrelevant columns if not needed
# df = df.drop(columns=["teacher_name"])

print("Cleaned data:")
print(df)
print()

# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------
# Features (X) = all columns except final_score
X = df.drop(columns=["final_score"])

# Target (y) = final_score we want to predict
y = df["final_score"]

# --------------------------------------------------
# 4. Split data into train and test
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --------------------------------------------------
# 5. Scale the features (important for some models)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 6. Train a regression model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# 7. Test the model
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# --------------------------------------------------
# 8. Make one new prediction
# --------------------------------------------------
new_student = pd.DataFrame({
    "hours_studied": [7],
    "previous_score": [80],
    "attendance": [88],
    "study_group": [1]
})

new_student_scaled = scaler.transform(new_student)
predicted_score = model.predict(new_student_scaled)[0]

print()
print(f"Predicted final score for a new student: {predicted_score:.2f}")