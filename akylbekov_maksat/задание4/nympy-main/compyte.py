from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Загрузка набора данных Breast Cancer
data = load_breast_cancer()

# Разделение датасета на обучающую и тестовую выборки с random_state=42
X_train_42, X_test_42, y_train_42, y_test_42 = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Разделение датасета на обучающую и тестовую выборки с random_state=1
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data.data, data.target, test_size=0.2, random_state=1)

# Проверка размеров обучающих и тестовых выборок
print("Размер обучающей выборки с random_state=42:", X_train_42.shape)
print("Размер тестовой выборки с random_state=42:", X_test_42.shape)
print("Размер обучающей выборки с random_state=1:", X_train_1.shape)
print("Размер тестовой выборки с random_state=1:", X_test_1.shape)