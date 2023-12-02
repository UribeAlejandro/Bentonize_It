import bentoml.mlflow
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def training_loop(X, y) -> DecisionTreeClassifier:
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        model_info = mlflow.sklearn.log_model(model, "model", input_example=X_train.loc[[0]])
        model_uri = model_info.model_uri

        eval_data = X_test
        eval_data["target"] = y_test

        mlflow.evaluate(model_uri, eval_data, targets="target", model_type="classifier", evaluators=["default"])

        bentoml.sklearn.save_model("iris_clf", model)

    return model
