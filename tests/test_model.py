import pytest
import numpy as np
import sys
import os

# Create a dummy model module for testing purposes
# In a real scenario, this would be your actual model module
DUMMY_MODEL_PY_CONTENT = """
import numpy as np
import time

class SimpleModel:
    def __init__(self, num_features=5, learning_rate=0.01, random_state=None):
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError("num_features must be a positive integer")
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = None
        self._is_trained = False
        self._random_state = random_state
        if self._random_state is not None:
            np.random.seed(self._random_state)

    def train(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        n_samples, n_features = X.shape
        if n_features != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {n_features}")
        if n_samples != len(y):
            raise ValueError("Number of samples in X and y do not match")

        # Simulate training process
        self.weights = np.random.rand(self.num_features) * self.learning_rate
        time.sleep(0.01) # Simulate training time
        self._is_trained = True

    def predict(self, X):
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        if not isinstance(X, np.ndarray):
             raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.num_features:
             raise ValueError(f"Expected {self.num_features} features, got {X.shape[1]}")

        # Simulate prediction (e.g., simple linear model + threshold)
        scores = X @ self.weights
        predictions = (scores > np.median(scores)).astype(int) # Example binary classification
        return predictions

    def evaluate(self, X, y):
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if X.shape[0] != len(y):
             raise ValueError("Number of samples in X and y do not match for evaluation")

        predictions = self.predict(X)
        # Simulate evaluation metric (e.g., accuracy)
        accuracy = np.mean(predictions == y)
        return {"accuracy": accuracy}

    @property
    def is_trained(self):
        return self._is_trained

"""

# Write the dummy model.py file to the current directory
with open("model.py", "w") as f:
    f.write(DUMMY_MODEL_PY_CONTENT)

# Now import the dummy model
try:
    from model import SimpleModel
except ImportError:
    # If running in an environment where the file isn't immediately discoverable
    # (like some CI/CD pipelines or isolated test runners), add cwd to path
    sys.path.insert(0, os.getcwd())
    from model import SimpleModel


# --- Test Fixtures ---

@pytest.fixture(scope="module")
def model_params():
    return {"num_features": 5, "learning_rate": 0.1, "random_state": 42}

@pytest.fixture
def untrained_model(model_params):
    """Provides an untrained model instance."""
    return SimpleModel(**model_params)

@pytest.fixture
def sample_data(model_params):
    """Provides sample training and testing data."""
    np.random.seed(model_params["random_state"]) # Ensure data consistency
    num_features = model_params["num_features"]
    n_train_samples = 100
    n_test_samples = 20

    X_train = np.random.rand(n_train_samples, num_features)
    y_train = (np.random.rand(n_train_samples) > 0.5).astype(int)

    X_test = np.random.rand(n_test_samples, num_features)
    y_test = (np.random.rand(n_test_samples) > 0.5).astype(int)

    return X_train, y_train, X_test, y_test

@pytest.fixture
def trained_model(untrained_model, sample_data):
    """Provides a trained model instance."""
    X_train, y_train, _, _ = sample_data
    untrained_model.train(X_train, y_train)
    return untrained_model


# --- Test Cases ---

def test_model_initialization(untrained_model, model_params):
    """Tests if the model initializes with correct parameters."""
    assert untrained_model.num_features == model_params["num_features"]
    assert untrained_model.learning_rate == model_params["learning_rate"]
    assert untrained_model.weights is None
    assert not untrained_model.is_trained
    assert untrained_model._random_state == model_params["random_state"]

def test_model_initialization_invalid_features():
    """Tests model initialization with invalid num_features."""
    with pytest.raises(ValueError, match="num_features must be a positive integer"):
        SimpleModel(num_features=0)
    with pytest.raises(ValueError, match="num_features must be a positive integer"):
        SimpleModel(num_features=-5)
    with pytest.raises(ValueError, match="num_features must be a positive integer"):
        SimpleModel(num_features=3.5)

def test_model_training(untrained_model, sample_data):
    """Tests the model training process."""
    X_train, y_train, _, _ = sample_data
    assert not untrained_model.is_trained
    assert untrained_model.weights is None

    untrained_model.train(X_train, y_train)

    assert untrained_model.is_trained
    assert isinstance(untrained_model.weights, np.ndarray)
    assert untrained_model.weights.shape == (untrained_model.num_features,)

def test_model_training_invalid_input_type(untrained_model):
    """Tests training with invalid input types."""
    X_list = [[1, 2], [3, 4]]
    y_array = np.array([0, 1])
    with pytest.raises(TypeError, match="X and y must be numpy arrays"):
        untrained_model.train(X_list, y_array)

def test_model_training_invalid_input_shape(untrained_model):
    """Tests training with invalid input shapes."""
    X_valid = np.random.rand(10, untrained_model.num_features)
    y_invalid_shape = np.random.rand(10, 1) # Should be 1D
    with pytest.raises(ValueError, match="y must be 1D"):
        untrained_model.train(X_valid, y_invalid_shape)

    X_invalid_dim = np.random.rand(10) # Should be 2D
    y_valid = np.random.rand(10)
    with pytest.raises(ValueError, match="X must be 2D"):
        untrained_model.train(X_invalid_dim, y_valid)


def test_model_training_feature_mismatch(untrained_model, sample_data):
    """Tests training with incorrect number of features."""
    _, y_train, _, _ = sample_data
    X_wrong_features = np.random.rand(len(y_train), untrained_model.num_features + 1)
    with pytest.raises(ValueError, match=f"Expected {untrained_model.num_features} features"):
        untrained_model.train(X_wrong_features, y_train)

def test_model_training_sample_mismatch(untrained_model, sample_data):
    """Tests training with mismatch in number of samples between X and y."""
    X_train, _, _, _ = sample_data
    y_wrong_samples = np.random.rand(len(X_train) + 1)
    with pytest.raises(ValueError, match="Number of samples in X and y do not match"):
        untrained_model.train(X_train, y_wrong_samples)

def test_model_prediction_before_training(untrained_model, sample_data):
    """Tests that prediction fails if the model is not trained."""
    _, _, X_test, _ = sample_data
    with pytest.raises(RuntimeError, match="Model must be trained before prediction."):
        untrained_model.predict(X_test)

def test_model_prediction(trained_model, sample_data):
    """Tests the model prediction output."""
    _, _, X_test, _ = sample_data
    predictions = trained_model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.ndim == 1
    assert len(predictions) == X_test.shape[0]
    assert np.issubdtype(predictions.dtype, np.integer) # Check if integer type
    assert set(np.unique(predictions)).issubset({0, 1}) # Check if binary output for this model

def test_model_prediction_invalid_input_type(trained_model):
    """Tests prediction with invalid input types."""
    X_list = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    with pytest.raises(TypeError, match="X must be a numpy array"):
        trained_model.predict(X_list)

def test_model_prediction_invalid_input_shape(trained_model):
    """Tests prediction with invalid input shapes."""
    X_invalid_dim = np.random.rand(trained_model.num_features) # Should be 2D
    with pytest.raises(ValueError, match="X must be 2D"):
        trained_model.predict(X_invalid_dim)

def test_model_prediction_feature_mismatch(trained_model, sample_data):
    """Tests prediction with incorrect number of features."""
    _, _, _, _ = sample_data
    X_wrong_features = np.random.rand(10, trained_model.num_features + 1)
    with pytest.raises(ValueError, match=f"Expected {trained_model.num_features} features"):
        trained_model.predict(X_wrong_features)

def test_model_evaluation_before_training(untrained_model, sample_data):
    """Tests that evaluation fails if the model is not trained."""
    _, _, X_test, y_test = sample_data
    with pytest.raises(RuntimeError, match="Model must be trained before evaluation."):
        untrained_model.evaluate(X_test, y_test)

def test_model_evaluation(trained_model, sample_data):
    """Tests the model evaluation output."""
    _, _, X_test, y_test = sample_data
    evaluation_results = trained_model.evaluate(X_test, y_test)

    assert isinstance(evaluation_results, dict)
    assert "accuracy" in evaluation_results
    accuracy = evaluation_results["accuracy"]
    assert isinstance(accuracy, (float, np.floating))
    assert 0.0 <= accuracy <= 1.0

def test_model_evaluation_invalid_input_type(trained_model):
    """Tests evaluation with invalid input types."""
    X_list = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    y_array = np.array([0, 1])
    with pytest.raises(TypeError, match="X and y must be numpy arrays"):
        trained_model.evaluate(X_list, y_array)

def test_model_evaluation_sample_mismatch(trained_model, sample_data):
    """Tests evaluation with mismatch in number of samples between X and y."""
    _, _, X_test, _ = sample_data
    y_wrong_samples = np.random.rand(len(X_test) + 5)
    with pytest.raises(ValueError, match="Number of samples in X and y do not match for evaluation"):
        trained_model.evaluate(X_test, y_wrong_samples)

# Clean up the dummy model file after tests run
def teardown_module(module):
    if os.path.exists("model.py"):
        os.remove("model.py")
    # Attempt to remove from sys.path if added
    if os.getcwd() in sys.path:
         try:
             sys.path.remove(os.getcwd())
         except ValueError:
             pass # Ignore if already removed or not present