# Bentoml It

## Description

This is a sample BentoML service. It was generated using [BentoML CLI](https://docs.bentoml.org/en/latest/reference/cli.html).

## Development

Install dependencies:

```bash
pip install -r requirements_dev.txt
```

Install the pre-commit hooks:

```bash
pre-commit install
```

If you are using a proxied MLFlow server, you can set the `MLFLOW_TRACKING_URI` environment variable to point to the server.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Training

The training loop saves the model to the bentoml repository. Then, to train the model, run the following command:

```bash
python -m src.main
```

After training the model, you can view the model in the bentoml repository:

```bash
bentoml list
```

On the other hand, the experiment tracking and model registry can be accessed through MLFlow.

## Serving

To serve the model locally, run the following command:

```bash
bentoml serve src/serving/service.py
```

## Deployment

Install the [operator](https://github.com/bentoml/bentoctl) of your choice, this case will be using the `google-cloud-run` operator.

```bash
bentoctl operator install google-cloud-run
```
