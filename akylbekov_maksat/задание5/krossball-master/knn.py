from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Загрузка набора данных Iris
iris = load_iris()
X, y = iris.data, iris.target

# Создание экземпляра классификатора kNN
knn = KNeighborsClassifier()

# Определение диапазона значений k, которые мы хотим проверить
k_values = list(range(1, 21))  # Проверяем значения k от 1 до 20

# Пустой список для хранения оценок производительности для каждого значения k
scores = []

# Цикл по каждому значению k
for k in k_values:
    # Выполнение кросс-валидации с k соседями
    knn.n_neighbors = k
    # Выполнение кросс-валидации с 5 фолдами, используя точность в качестве метрики оценки
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    # Усреднение оценок производительности для данного значения k и добавление в список
    scores.append(cv_scores.mean())

# Выводим результаты
for k, score in zip(k_values, scores):
    print(f"k = {k}, Средняя точность: {score:.4f}")
