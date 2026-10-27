from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# House data: [size_in_sqft, bedrooms], price
data = [
    [1200, 2, 180000],
    [1500, 3, 220000],
    [1700, 3, 250000],
    [2000, 4, 310000],
    [2300, 4, 360000],
    [2600, 5, 420000],
    [3000, 5, 480000],
    [3500, 6, 560000],
]

# Split into features X and target y
X = [[row[0], row[1]] for row in data]
y = [row[2] for row in data]

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train a simple regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Show results
print("Predicted prices for the test data:")
for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual} kr | Predicted: {predicted:.0f} kr")

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Absolute Error: {mae:.0f} kr")
print(f"R² score: {r2:.2f}")

# Make a new prediction
new_house = [[2500, 4]]
predicted_price = model.predict(new_house)[0]

print(f"\nPrediction for a new house: {predicted_price:.0f} kr")