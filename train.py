
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()

# Separate inputs and expected answers
X = iris.data
y = iris.target

# Create the machine learning model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Model trained and saved successfully!")