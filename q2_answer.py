from matplotlib import pyplot as plt
import numpy as np
from q2_script import read_dataset

from logistic_regression import LogisticRegression, WeightSelectionStrategy

X_train, y_train, X_test, y_test = read_dataset()


def split_dataset(split_index: int):
    X_validation = X_train[:split_index]
    y_validation = y_train[:split_index]
    X_training = X_train[split_index:]
    y_training = y_train[split_index:]
    return X_training, y_training, X_validation, y_validation


X_training, y_training, X_validation, y_validation = split_dataset(10000)


def question_2_1():
    print("Question 2.1")
    logistic_regression = LogisticRegression(
        X_training=X_training, y_training=y_training
    )
    logistic_regression.train(X_validation, y_validation)

    logistic_regression.test(X_test, y_test)


def plot_results(labels: list[str], accuracies: list[list[float]], plot_title: str):
    # label the axes
    plt.figure(figsize=(10, 10))
    for i in range(len(labels)):
        plt.plot(accuracies[i], label=labels[i])
    plt.legend()
    plt.title(plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


def get_different_batch_size_results() -> tuple[list[int], list[list[float]]]:
    batch_sizes = [1, 64, 50000]
    accuracies = []
    for batch_size in batch_sizes:
        print(f"Running with batch size {batch_size}")
        logistic_regression = LogisticRegression(
            X_training=X_training, y_training=y_training, batch_size=batch_size
        )
        accuracies.append(logistic_regression.train(X_validation, y_validation))

    return batch_sizes, accuracies


def get_different_weight_strategy_results() -> tuple[list[str], list[list[float]]]:
    weight_strategies = [
        WeightSelectionStrategy.UNIFORM,
        WeightSelectionStrategy.ZERO,
        WeightSelectionStrategy.NORMAL,
    ]
    accuracies = []
    for weight_strategy in weight_strategies:
        print(f"Running with weight strategy {weight_strategy}")
        logistic_regression = LogisticRegression(
            X_training=X_training,
            y_training=y_training,
            weight_strategy=weight_strategy,
        )
        accuracies.append(logistic_regression.train(X_validation, y_validation))

    return weight_strategies, accuracies


def get_different_regularization_coefficient_results() -> (
    tuple[list[float], list[list[float]]]
):
    regularization_coefficients = [10**-2, 10**-4, 10**-9]
    accuracies = []
    for regularization_coefficient in regularization_coefficients:
        print(f"Running with regularization coefficient {regularization_coefficient}")
        logistic_regression = LogisticRegression(
            X_training=X_training,
            y_training=y_training,
            reqularization_coeficient=regularization_coefficient,
        )
        accuracies.append(logistic_regression.train(X_validation, y_validation))

    return regularization_coefficients, accuracies


def get_different_learning_rate_results() -> tuple[list[float], list[list[float]]]:
    learning_rates = [0.1, 10**-3, 10**-4, 10**-5]
    accuracies = []
    for learning_rate in learning_rates:
        print(f"Running with learning rate {learning_rate}")
        logistic_regression = LogisticRegression(
            X_training=X_training,
            y_training=y_training,
            learning_rate=learning_rate,
        )
        accuracies.append(logistic_regression.train(X_validation, y_validation))

    return learning_rates, accuracies


def plot_different_batch_size_results():
    print("Plotting different batch size results")
    batch_sizes, batch_size_accuracies = get_different_batch_size_results()

    batch_size_labels = [f"Batch size: {batch_size}" for batch_size in batch_sizes]
    plot_results(
        labels=batch_size_labels,
        accuracies=batch_size_accuracies,
        plot_title="Batch size vs accuracy",
    )


def plot_different_weights_strategy_results():
    print("Plotting different weight strategy results")
    (
        weight_strategies,
        weight_strategy_accuracies,
    ) = get_different_weight_strategy_results()

    weight_strategy_labels = [
        f"Weight strategy: {weight_strategy.value}"
        for weight_strategy in weight_strategies
    ]
    plot_results(
        labels=weight_strategy_labels,
        accuracies=weight_strategy_accuracies,
        plot_title="Weight strategy vs accuracy",
    )


def plot_different_regularization_coefficient_results():
    print("Plotting different regularization coefficient results")
    (
        regularization_coefficients,
        regularization_coefficient_accuracies,
    ) = get_different_regularization_coefficient_results()

    regularization_coefficient_labels = [
        f"Regularization coefficient: {regularization_coefficient}"
        for regularization_coefficient in regularization_coefficients
    ]
    plot_results(
        labels=regularization_coefficient_labels,
        accuracies=regularization_coefficient_accuracies,
        plot_title="Regularization coefficient vs accuracy",
    )


def plot_different_learning_rate_results():
    print("Plotting different learning rate results")
    (
        learning_rates,
        learning_rate_accuracies,
    ) = get_different_learning_rate_results()

    learning_rate_labels = [
        f"Learning rate: {learning_rate}" for learning_rate in learning_rates
    ]
    plot_results(
        labels=learning_rate_labels,
        accuracies=learning_rate_accuracies,
        plot_title="Learning rate vs accuracy",
    )

def visualize_weights(weights: np.ndarray):
    n = weights.shape[1]  # Number of weights
    cols = 4  # Define the number of columns for your grid
    rows = n // cols  # Define the number of rows for your grid
    rows += n % cols  # Add an extra row if there are any remaining plots

    pos = range(1, n + 1)

    fig = plt.figure(figsize=(20, 20))

    for k, weight in zip(pos, weights.T):
        ax = fig.add_subplot(rows, cols, k)
        ax.matshow(weight.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5*weights.min(), vmax=0.5*weights.max())

    plt.show()


def question_2_2():
    print("Question 2.2")

    plot_different_batch_size_results()
    plot_different_weights_strategy_results()
    plot_different_regularization_coefficient_results()
    plot_different_learning_rate_results()



def question_2_345():
    print("Questions 2.3, 2.4, 2.5")
    weight_selection_strategy = WeightSelectionStrategy.ZERO
    learning_rate = 10**-3
    batch_size = 64
    regularization_coefficient = 10**-2

    logistic_regression = LogisticRegression(
        X_training=X_training,
        y_training=y_training,
        weight_strategy=weight_selection_strategy,
        learning_rate=learning_rate,
        batch_size=batch_size,
        reqularization_coeficient=regularization_coefficient,
    )
    logistic_regression.train(X_validation, y_validation)

    logistic_regression.test(X_test, y_test)

    weights = logistic_regression.get_weights()
    
    visualize_weights(weights)
    


def question_2():
    question_2_1()
    question_2_2()
    question_2_345()
