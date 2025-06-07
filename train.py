import bentoml

from sklearn import svm
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Save the model to the BentoML model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")

## iris_clf:au57ebcdmokfttch