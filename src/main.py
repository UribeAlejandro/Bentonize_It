from src.model.training import training_loop
from src.preprocess.etl import extract_data, transform_data

if __name__ == "__main__":
    data = extract_data()
    X, y = transform_data(data)
    training_loop(X, y)
