from bentoml import HTTPServer


def test_service():
    server = HTTPServer("iris_clf:latest", production=True, port=3000, host="0.0.0.0")

    # Start the server (non-blocking by default)
    server.start(blocking=False)

    # Get a client to make requests to the server
    client = server.get_client()

    # Send a request using the client
    result = client.classify({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})

    # Stop the server to free up resources
    server.stop()

    assert result["label"] in ["setosa", "versicolor", "virginica"]
