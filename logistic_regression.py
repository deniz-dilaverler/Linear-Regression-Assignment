from enum import Enum
import numpy as np
from confusion_matrix import ConfusionMatrix


def get_accuracy(predictions: np.ndarray, actual_labels: np.ndarray):
    actual_labels = np.argmax(actual_labels, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(actual_labels == predicted_labels)

    return accuracy


def get_confusion_matrix(
    predictions: np.ndarray, actual_labels: np.ndarray
) -> ConfusionMatrix:
    actual_labels = np.argmax(actual_labels, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    confusion_matrix = ConfusionMatrix(10, 10)
    confusion_matrix.add_predictions_batch(predicted_labels, actual_labels)

    return confusion_matrix


class WeightSelectionStrategy(Enum):
    GAUSIAN = "gausian"
    UNIFORM = "uniform"
    ZERO = "zero"
    NORMAL = "normal"


class LogisticRegression:
    EPOCHS: int = 100
    learning_rate: float
    batch_size: int
    reqularization_coeficient: float
    __biases: np.ndarray
    __weights: np.ndarray
    __X_training: np.ndarray
    __y_training: np.ndarray

    def __init__(
        self,
        X_training: np.ndarray,
        y_training: np.ndarray,
        learning_rate: float = 5 * 10**-4,
        batch_size: int = 200,
        reqularization_coeficient: float = 10**-4,
        weight_strategy: WeightSelectionStrategy = WeightSelectionStrategy.GAUSIAN,
    ):
        self.__y_training = y_training
        self.__X_training = X_training
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reqularization_coeficient = reqularization_coeficient
        self.__biases = np.zeros(self.__class_count)
        self.__initialize_weights(weight_strategy)

    def __initialize_weights(self, weight_strategy: WeightSelectionStrategy):
        if weight_strategy == WeightSelectionStrategy.GAUSIAN:
            self.__weights = np.random.normal(
                0, 1, size=(self.__feature_count, self.__class_count)
            )
        elif weight_strategy == WeightSelectionStrategy.UNIFORM:
            self.__weights = np.random.uniform(
                low=0, high=1, size=(self.__feature_count, self.__class_count)
            )
        elif weight_strategy == WeightSelectionStrategy.ZERO:
            self.__weights = np.zeros((self.__feature_count, self.__class_count))
        elif weight_strategy == WeightSelectionStrategy.NORMAL:
            self.__weights = np.random.normal(
                0, 1, size=(self.__feature_count, self.__class_count)
            )
        else:
            raise Exception("Invalid weight selection strategy")

    def get_weights(self):
        return self.__weights
    
    @property
    def __feature_count(self):
        return self.__X_training.shape[1]

    @property
    def __class_count(self):
        return self.__y_training.shape[1]

    @staticmethod
    def __softmax(z: int):
        exp_z = np.exp(z)
        exp_z_sum = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z / exp_z_sum

    def predict(self, X: np.ndarray):
        logits = np.dot(X, self.__weights) + self.__biases
        return self.__softmax(logits)

    def __train_epoch(self):
        for i in range(0, len(self.__X_training), self.batch_size):
            X_batch = self.__X_training[i : i + self.batch_size]
            y_batch = self.__y_training[i : i + self.batch_size]

            prediction = self.predict(X_batch)
            loss = np.sum(prediction - y_batch, axis=0)
            self.__weights -= (
                np.dot(X_batch.T, prediction - y_batch)
                + self.reqularization_coeficient * self.__weights
            ) * self.learning_rate
            self.__biases -= self.learning_rate * loss

    def __evaluate_on_validation_set(
        self, X_validation: np.ndarray, y_validation: np.ndarray
    ):
        predictions = self.predict(X_validation)
        accuracy = get_accuracy(predictions, y_validation)
        return accuracy

    def train(
        self,
        X_validation: int,
        y_validation: int,
    ) -> list[float]:
        accuracies = []
        for epoch in range(self.EPOCHS):
            self.__train_epoch()
            accuracy = self.__evaluate_on_validation_set(
                X_validation=X_validation, y_validation=y_validation
            )
            accuracies.append(accuracy)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} accuracy: {accuracy}")
        return accuracies

    def test(self, X_test: np.ndarray, y_test: np.ndarray):
        predictions = self.predict(X_test)
        accuracy = get_accuracy(predictions, y_test)

        confusion_matrix = get_confusion_matrix(predictions, y_test)

        print(f"Test Accuracy: {accuracy}")

        print("Confusion Matrix:")
        print(confusion_matrix)

        print("Metrics:")
        confusion_matrix.print_metrics()

        return accuracy
