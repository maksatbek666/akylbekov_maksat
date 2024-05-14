import numpy as np

def euclidean_distance(x1, x2):
    """
    Вычисляет евклидово расстояние между двумя точками.

    Параметры:
    x1: numpy.ndarray
        Первая точка.
    x2: numpy.ndarray
        Вторая точка.

    Возвращает:
    float
        Евклидово расстояние между точками.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """
    Вычисляет манхэттенское расстояние между двумя точками.

    Параметры:
    x1: numpy.ndarray
        Первая точка.
    x2: numpy.ndarray
        Вторая точка.

    Возвращает:
    float
        Манхэттенское расстояние между точками.
    """
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    """
    Вычисляет расстояние Чебышёва между двумя точками.

    Параметры:
    x1: numpy.ndarray
        Первая точка.
    x2: numpy.ndarray
        Вторая точка.

    Возвращает:
    float
        Расстояние Чебышёва между точками.
    """
    return np.max(np.abs(x1 - x2))



