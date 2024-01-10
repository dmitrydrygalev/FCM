import csv
import io
import json

from helpers import parse_vector

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skfuzzy as fuzz

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
cast_data.drop(['ID', 'Время создания', 'В какой тематике вы бы хотели участвовать в научной конференции?',
                'Учавствовали ли вы в конференциях    1548428932'], axis=1, inplace=True)  # удаление не нужных столбцов
cast_data.sort_values(by='Выберите свою группу', inplace=True)  # сортировка всех строк по группам
cast_data.drop(['Выберите свою группу'], axis=1, inplace=True)

cast_data.to_csv("res.csv",
                 index=False)

with open('weights.json', 'r', encoding='utf-8') as file:  # чтение json файла с весами
    weights = json.load(file)

cach_key = ""  # переменная для кеширования ключа словаря

for column in cast_data:  # перебор стобцов в датафрейме
    for index, value in cast_data[column].items():  # перебор всех значений столбца
        for key in weights.keys():  # поиск столбца в ключах словаря
            if key == column:  # если ключ из словаря совпал с названием текущего столбца, то кешируется ключ и прекрывается цикл
                cach_key = key
                break
        for dict_item in weights[cach_key]:  # замена данных в датафрейме, на данные словаря (векторизация)
            if dict_item == value:  # если признак словаря и значение столбца совпадают, то значение столбца перезаписывается на значение признака словаря
                cast_data.at[index, column] = weights[cach_key][dict_item]

cast_data.to_csv("vector_cast_data.csv",
                 index=False)  # запись нового csv файла, с данными переведенными в векторный вид и отсоритрованными по группам

cast_data = cast_data.reset_index(drop=True)

fourth_kurse = cast_data.iloc[27:29]

one_array = []
i = 0

for col in fourth_kurse.columns:
    i += 1
    summed_vector = [0, 0, 0, 0, 0]
    for index, value in fourth_kurse[col].items():
        summed_vector = [summed_vector[i] + value[i] for i in range(len(summed_vector))]
    one_array.append(summed_vector)


plt.imshow(one_array, cmap='magma', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.show()


for index, row in fourth_kurse.iterrows():
    for column in fourth_kurse.columns:
        value = row[column]
        new_value = parse_vector(value)
        row[column] = new_value


only_values = fourth_kurse.values.T
only_values = only_values.astype(float)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    only_values, 2, 1.1, error=0.005, maxiter=1000, init=None
)

cluster_membership = np.argmax(u, axis=0)


fourth_kurse['Cluster'] = cluster_membership + 1

fig, axes = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=True)

for i in range(3):
    for j in range(5):
        idx = i * 5 + j
        axes[i, j].scatter(fourth_kurse.iloc[:, idx], fourth_kurse['Cluster'], c=fourth_kurse['Cluster'], cmap='viridis', s=30)
        axes[i, j].set_title(f'Question {idx+1}')

plt.suptitle('Fuzzy C-means Clustering of Conference Interest')
plt.show()










