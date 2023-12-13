import numpy as np


# rows are predictions
# columns are actual
class ConfusionMatrix:
    matrix: np.ndarray

    def __init__(self, dim_x: int, dim_y: int):
        self.matrix = np.zeros((dim_x, dim_y), dtype=int)

    def add_prediction(self, predicted: int, actual: int):
        self.matrix[predicted][actual] += 1

    def add_predictions_batch(self, predictions: np.ndarray, actuals: np.ndarray):
        for i in range(len(predictions)):
            self.add_prediction(predictions[i], actuals[i])

    def __get_precision(self, class_index: int) -> float:
        true_positive = self.matrix[class_index][class_index]
        false_positive = np.sum(self.matrix[:, class_index]) - true_positive
        if true_positive + false_positive == 0:
            return 0

        return true_positive / (true_positive + false_positive)

    def __get_recall(self, class_index: int) -> float:
        true_positive = self.matrix[class_index][class_index]
        false_negative = np.sum(self.matrix[class_index, :]) - true_positive
        if true_positive + false_negative == 0:
            return 0

        return true_positive / (true_positive + false_negative)

    def get_recalls(self) -> list[float]:
        recalls = []
        for i in range(len(self.matrix)):
            recalls.append(self.__get_recall(i))
        return recalls

    def get_precisions(self) -> list[float]:
        precisions = []
        for i in range(len(self.matrix)):
            precisions.append(self.__get_precision(i))
        return precisions

    def __get_f1_score(self, class_index: int) -> float:
        precision = self.__get_precision(class_index)
        recall = self.__get_recall(class_index)
        if precision + recall == 0:
            return 0

        return 2 * (precision * recall) / (precision + recall)

    def __get_f2_score(self, class_index: int) -> float:
        precision = self.__get_precision(class_index)
        recall = self.__get_recall(class_index)
        if precision + recall == 0:
            return 0

        return 5 * (precision * recall) / (4 * precision + recall)

    def get_f1_scores(self) -> list[float]:
        f1_scores = []
        for i in range(len(self.matrix)):
            f1_scores.append(self.__get_f1_score(i))
        return f1_scores


    def get_f2_scores(self) -> list[float]:
        f2_scores = []
        for i in range(len(self.matrix)):
            f2_scores.append(self.__get_f2_score(i))
        return f2_scores
    
    def __str__(self):
        return str(self.matrix)
    
    def print_metrics(self):
        # print metrics for each class
        for i in range(len(self.matrix)):
            print(f"Class {i}")
            print(f"Recall: {self.__get_recall(i)}")
            print(f"Precision: {self.__get_precision(i)}")
            print(f"F1 Score: {self.__get_f1_score(i)}")
            print(f"F2 Score: {self.__get_f2_score(i)}")
            print()
