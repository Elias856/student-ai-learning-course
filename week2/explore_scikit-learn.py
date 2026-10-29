# Lesson 9: Explore scikit-learn and learn how to find the right tools
# Students should open the links below and read the docs to understand each part.
#
# Read these pages:
# https://scikit-learn.org/stable/modules/classes.html
#   You se all the classes. In the left menu, you can se the classes in sections. A good place to cearch
# https://scikit-learn.org/stable/datasets/toy_dataset.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

# --------------------------------------------------
# 1. Import the needed tools from scikit-learn
# --------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 2. Load a small built-in dataset
# --------------------------------------------------
# Students: Find out what this dataset contains in the scikit-learn docs
iris = load_iris()

X = iris.data
y = iris.target

print("Step 2: Dataset loaded")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Class names:", iris.target_names)

# --------------------------------------------------
# 3. Look at the first rows of the data
# --------------------------------------------------
# Students: Search the docs to understand what the data represents
print()
print("Step 3: First rows of the dataset")
for i in range(5):
    print(X[i], "-> class:", y[i])

# --------------------------------------------------
# 4. Split the data into training and testing data
# --------------------------------------------------
# Students: Read the train_test_split documentation and find out
# what test_size and random_state do
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print()
print("Step 4: Data split finished")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# --------------------------------------------------
# 5. Choose a model from scikit-learn
# --------------------------------------------------
# Students: Use the scikit-learn website to find a model
# for classification. LogisticRegression is a good beginner choice.
model = LogisticRegression(max_iter=200)

print()
print("Step 5: Model chosen:", type(model).__name__)

# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------
model.fit(X_train, y_train)

print()
print("Step 6: Model trained")

# --------------------------------------------------
# 7. Make predictions on the test data
# --------------------------------------------------
y_pred = model.predict(X_test)

print()
print("Step 7: Predictions made")
print("First 10 predictions:", y_pred[:10])

# --------------------------------------------------
# 8. Evaluate the model
# --------------------------------------------------
# Students: Look up accuracy_score in the docs
accuracy = accuracy_score(y_test, y_pred)

print()
print("Step 8: Model accuracy")
print("Accuracy:", round(accuracy, 3))

# --------------------------------------------------
# 9. Make one new prediction
# --------------------------------------------------
# Students: Look up how to make one prediction with a trained model
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
predicted_class = iris.target_names[prediction[0]]

print()
print("Step 9: Prediction for a new flower")
print("New flower:", new_flower)
print("Predicted class:", predicted_class)