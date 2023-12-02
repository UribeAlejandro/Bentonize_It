from sklearn.datasets import load_iris


def extract_data():
    iris = load_iris(as_frame=True)
    return iris


def transform_data(data):
    X = data.data
    y = data.target

    X = X.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    return X, y


def load_data(data):
    pass
