fourth_kurse = cast_data.head(26)

array = fourth_kurse.values
one_array = []
i = 0

for col in fourth_kurse.columns:
    i += 1
    summed_vector = [0, 0, 0, 0, 0]
    for index, value in fourth_kurse[col].items():
        summed_vector = [summed_vector[i] + value[i] for i in range(len(summed_vector))]
    one_array.append(summed_vector)


data_matrix = np.array(one_array)

# Транспонирование матрицы, чтобы строки представляли признаки, а столбцы - данные
data_matrix = data_matrix.T

print(data_matrix)

# Применение алгоритма C-Means с помощью библиотеки scikit-fuzzy
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_matrix, 5, 1.8, error=0.005, maxiter=1000, init=None)

# Вывод результатов
cluster_membership = np.argmax(u, axis=0)  # Определение принадлежности каждого элемента к кластеру

height_weight = []
index_HW = 0

for i in range(len(one_array)):
    height_weight.append(one_array[i][0] + one_array[i][1])
    print("Данные ответов:", one_array[i])
    print("Кластер:", cluster_membership[i])
    print("Принадлежность к кластерам:", u[:, i])
    index_HW = height_weight.index(max(height_weight))
    print()

inclined = cluster_membership[index_HW]

inclined_degree = 0

for i in range(len(one_array)):
    if cluster_membership[i] == inclined:
        inclined_degree += 1


print(inclined)


plt.imshow(one_array, cmap='magma', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.show()


coordinates = []
fuzz_coordinates = []
col = 0

for column in cast_data:  # перебор стобцов в датафрейме
    col += 1
    for index, value in cast_data[column].items():  # перебор всех значений столбца
        coordinates.append({"y": col, "x": parse_vector(value)})
        fuzz_coordinates.append([parse_vector(value), col])

x_values = [coord['x'] for coord in coordinates]
y_values = [coord['y'] for coord in coordinates]

# Строим график
plt.scatter(x_values, y_values, color='b', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Координаты на плоскости')

plt.show()




# original_FC = fuzz_coordinates
fuzz_coordinates_NP = np.array(fuzz_coordinates)
fuzz_coordinates = fuzz_coordinates_NP.T

question_numbers = fuzz_coordinates[:, 0]
answer_weights = fuzz_coordinates[:, 1]

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(fuzz_coordinates, 2, 1.8, error=0.005, maxiter=1000, init=None)

cluster_membership = np.argmax(u, axis=0)

plt.scatter(question_numbers, answer_weights, c=cluster_membership)
plt.show()

#for obj in coordinates:
#    print(f++"\n{obj}")















only_values = fourth_kurse.values.T

cntr, u, _, _, _, _,  = fuzz.cluster.cmeans(
    only_values, 3, 2, error=0.005, maxiter=1000, init=None
)

cluster_membership = np.argmax(u, axis=0)


fig, axes = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=True)

for i in range(3):
    for j in range(5):
        idx = i * 5 + j
        axes[i, j].scatter(cast_data.iloc[:, idx], cluster_membership, c=cluster_membership, cmap='viridis', s=30)
        axes[i, j].set_title(f'Question {idx+1}')

plt.suptitle('Fuzzy C-means Clustering of Conference Interest')
plt.show()
