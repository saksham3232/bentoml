import bentoml
import numpy as np

# Load the actual scikit-learn model (not the model reference)
model = bentoml.sklearn.load_model("iris_clf:latest")

# Example input for prediction
input_data = np.array([[5.9, 3.0, 5.1, 1.8]])
predictions = model.predict(input_data)

print(predictions)  # Output: [2]