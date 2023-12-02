from typing import Literal

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
svc = bentoml.Service("iris_clf", runners=[iris_clf_runner])


class Request(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Response(BaseModel):
    label: Literal["setosa", "versicolor", "virginica"]


@svc.api(input=JSON(pydantic_model=Request), output=JSON(pydantic_model=Response))
def classify(request: Request) -> Response:
    input_ = [
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]

    label_idx = iris_clf_runner.predict.run([input_])[0]
    label = ["setosa", "versicolor", "virginica"][label_idx]

    return Response(label=label)
