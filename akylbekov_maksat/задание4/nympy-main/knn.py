from typing import Callable
from distances import euclidean_distance, manhattan_distance, chebyshev_distance

class KNNClassifier:
    def __init__(self, k: int, distances: Callable = euclidean_distance):
        self.k = k
        self.distances = distances

    def fit(self, x_train: list, y_train: list):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: list) -> list:
        y_pred = []
        for xi_test in x_test:
            dist_mark = []
            for xi_train, yi_train in zip(self.x_train, self.y_train):
                dist = self.distances(xi_train, xi_test)
                dist_mark.append((dist, yi_train))
            dist_mark.sort(key=lambda x: x[0])
            dist_mark = dist_mark[:self.k]
            mark = [m for _, m in dist_mark]
            mark_count = {m: mark.count(m) for m in mark}
            result = max(mark_count, key=mark_count.get)
            y_pred.append(result)
        return y_pred
