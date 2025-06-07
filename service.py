import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Load the model runner
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Define the BentoML service
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# Define the API endpoint using the updated syntax
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await iris_clf_runner.async_run(input_series)