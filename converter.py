import json
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import io

initial_data = pd.read_csv("data.csv")  # чтение csv файла

row_data = initial_data.values  # получение содержимового только строк

column_names_series = initial_data.iloc[0]  # получение первой строки, в которой содержаться навзания всех столбцов

column_names_string = column_names_series.to_string()  # преобразование полученной строки из series в string

column_names_cvs_string = csv.reader(io.StringIO(column_names_string),
                                     delimiter=',')  # преобразование строки с названиями колонок в массив
columns_array = []  # инициализация переменной

for columns in column_names_cvs_string:  # запись данных из csv в массив
    columns_array = columns

columns_array = columns_array[:-3]  # удаление последних трех значений в массиве

rows = []  # инициализация массива который содержит строки для создания нового датафрейма

for row in row_data:  # берется каждый массив в массиве строк
    for item in row:  # берется строка в массиве
        reader = csv.reader(io.StringIO(item),
                            delimiter=',')  # разделение строки по запятым, с учетом ковычек(если слова в ковычках, они не раздяляеются запятыми)
        for string in reader:  # берется каждая строка в csv
            rows.append(string)  # запись массива со строкой в массив

cast_data = pd.DataFrame(rows, columns=columns_array)  # создание нового датафрейма с названиями и столбцами из data.csv
cast_data.drop(['ID', 'Время создания', 'В какой тематике вы бы хотели участвовать в научной конференции?', 'Учавствовали ли вы в конференциях    1548428932'], axis=1, inplace=True)  # удаление не нужных столбцов
cast_data.sort_values(by='Выберите свою группу', inplace=True) # сортировка всех строк по группам
cast_data.drop(['Выберите свою группу'], axis=1, inplace=True)

print(type(cast_data))

# Загрузка весов из JSON файла
with open('weights.json', 'r', encoding='utf-8') as file:
    weights = json.load(file)

cash_key = ""  # переменная для кеширования ключа словаря

for column in cast_data:  # перебор стобцов в датафрейме
    for index, value in cast_data[column].items():  # перебор всех значений столбца
        for key in weights.keys():  # поиск столбца в ключах словаря
            if key == column:  # если ключ из словаря совпал с названием текущего столбца, то кешируется ключ и прекрывается цикл
                cash_key = key
                break
        for dict_item in weights[cash_key]:  # замена данных в датафрейме, на данные словаря (векторизация)
            if dict_item == value: # если признак словаря и значение столбца совпадают, то значение столбца перезаписывается на значение признака словаря
                cast_data.at[index, column] = weights[cash_key][dict_item]

# Убедимся, что все данные численные
cast_data = cast_data.apply(pd.to_numeric, errors='coerce')
cast_data = cast_data.dropna()


# Кластеризация c-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    cast_data.T, 3, 2, error=0.005, maxiter=1000)

# Уменьшение до 2 компонент с помощью PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(cast_data)

# Получение центров кластеров в оригинальном пространстве из результата fuzzy c-means
cluster_centers = pca.transform(cntr)

# Визуализация кластеров
plt.figure()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.argmax(u, axis=0))
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', label='Centroids')
plt.title("Кластеризация с помощью fuzzy c-means и PCA")
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.legend()
plt.show()

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(cast_data.T, 3, 2, error=0.005, maxiter=1000)

pca = PCA(n_components=2)  # Уменьшение до 2 компонент
reduced_data = pca.fit_transform(cast_data.astype(np.float64))  # Преобразование исходных данных

# Получение центров кластеров в оригинальном пространстве из результата fuzzy c-means
cluster_centers = pca.transform(cntr)

# Визуализация кластеров
plt.figure()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.argmax(u, axis=0))  # Аргумент c - метки кластеров
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', label='Centroids')  # Центроиды кластеров
plt.title("Кластеризация с помощью fuzzy c-means и PCA")
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.legend()
plt.show()



final_vectors = []

for index, row in cast_data.iterrows():
    # Итерация по каждой колонке в строке
    summed_vector = [0, 0, 0, 0, 0]
    for column in cast_data.columns:
        # Получение значения элемента
        value = row[column]
        summed_vector = [summed_vector[i] + value[i] for i in range(len(summed_vector))]
    final_vectors.append(summed_vector)



fig, ax = plt.subplots()

# Перебор каждого внутреннего массива
for vector in final_vectors:
    x, y, z, a, b = vector

    # Отображение точки на плоскости
    ax.plot(x, y, marker='o', markersize=5, color='blue')

# Отображение числовой плоскости
plt.grid(True)
plt.show()




n_clusters = 2
cmeans = KMeans(n_clusters=n_clusters)

# Обучаем модель на данных
cmeans.fit(cast_data)

centers = cmeans.cluster_centers_

# Получаем метки принадлежности кластерам
labels = cmeans.labels_

cluster_labels = np.argmin(np.linalg.norm(cast_data[:, np.newaxis] - centers, axis=-1), axis=-1)

# Определяем результат для каждой записи
results = ['Склонен участвовать' if label == 1 else 'Не склонен участвовать' for label in cluster_labels]



n_a = np.array(one_array)

n_clusters = 2
cmeans = KMeans(n_clusters=n_clusters)

# Обучаем модель на данных
cmeans.fit(n_a)

centers = cmeans.cluster_centers_

# Получаем метки принадлежности кластерам
labels = cmeans.labels_

cluster_labels = np.argmin(np.linalg.norm(n_a[:, np.newaxis] - centers, axis=-1), axis=-1)

# Определяем результат для каждой записи
results = ['Склонен участвовать' if label == 1 else 'Не склонен участвовать' for label in cluster_labels]

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
n_a = np.vstack(one_array)

print(n_a)

n_samples = 1500
n_centers = 3

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    n_a.T, n_centers, 2, error=0.005, maxiter=1000, init=None)

# Построим результат кластеризации
# Вычисляем принадлежность к каждому кластеру
cluster_membership = np.argmax(u, axis=0)

# Визуализация результатов
colors = ['b', 'orange', 'g', 'r']
fig, ax0 = plt.subplots()
for j in range(n_centers):
    ax0.plot(n_a[cluster_membership == j][:,0],
             n_a[cluster_membership == j][:,1], '.', color=colors[j])

for pt in cntr:
    ax0.plot(pt[0], pt[1], 'rs')

plt.title('Центроиды и данные алгоритма Fuzzy C-means')
plt.show()



n_a = np.vstack(one_array)

n_centers = 3

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    n_a.T, n_centers, 2, error=0.005, maxiter=1000, init=None)

# Построим результат кластеризации
# Вычисляем принадлежность к каждому кластеру
cluster_membership = np.argmax(u, axis=0)

# Визуализация результатов
colors = ['b', 'orange', 'g', 'r']
fig, ax0 = plt.subplots()
for j in range(n_centers):
    ax0.plot(n_a[cluster_membership == j][:, 0],
             n_a[cluster_membership == j][:, 1], '.', color=colors[j])

for pt in cntr:
    ax0.plot(pt[0], pt[1], 'rs')

plt.title('Центроиды и данные алгоритма Fuzzy C-means')
plt.show()

n_centers = 2

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    n_a.T, n_centers, 2, error=0.005, maxiter=1000, init=None)

# Построим результат кластеризации
# Вычисляем принадлежность к каждому кластеру
cluster_membership = np.argmax(u, axis=0)

# Визуализация результатов
colors = ['b', 'orange', 'g', 'r']
fig, ax0 = plt.subplots()
for j in range(n_centers):
    ax0.plot(n_a[cluster_membership == j][:, 0],
             n_a[cluster_membership == j][:, 1], '.', color=colors[j])


for pt in cntr:
    ax0.plot(pt[0], pt[1], 'rs')

plt.title('Центроиды и данные алгоритма Fuzzy C-means')
plt.show()
