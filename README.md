# BentoML Iris Classifier

A minimal example of using BentoML to build and serve a machine learning model for iris flower classification with scikit-learn.

## Features

- Trains an SVM classifier on the classic Iris dataset.
- Saves and loads models using BentoML's model store.
- Exposes a service endpoint for prediction using BentoML's Service API.
- Async API for fast, scalable inference.

## Setup

### Prerequisites

- Python 3.7+
- Recommended to use a virtual environment

### Install dependencies

```bash
pip install bentoml scikit-learn numpy
```

## Training the Model

Train and save the classifier:

```bash
python train.py
```

This will:
- Load the Iris dataset
- Train an SVM classifier
- Save the trained model to the BentoML model store

## Running the Service

The BentoML service exposes an API for iris flower classification.

1. Ensure the model is trained and saved (see above).
2. Start the BentoML service:

```python
# service.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await iris_clf_runner.async_run(input_series)
```

3. Serve with BentoML:

```bash
bentoml serve service:svc
```

## Example Usage

### Predict with Saved Model

```python
import bentoml
import numpy as np

model = bentoml.sklearn.load_model("iris_clf:latest")
input_data = np.array([[5.9, 3.0, 5.1, 1.8]])
predictions = model.predict(input_data)
print(predictions)  # Output: [2]
```

### Test API Endpoint

Send a numpy array to the API endpoint for classification.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## License

This project is for educational purposes and does not currently specify a license.

## Acknowledgments

- [BentoML](https://bentoml.com/)
- [scikit-learn](https://scikit-learn.org/)
