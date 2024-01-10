import numpy as np
import skfuzzy as fuzz
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(42)
data = np.random.randint(1, 6, size=(170, 15))  # Пример: случайные значения от 1 до 5


columns = [f'Question_{i+1}' for i in range(15)]
df = pd.DataFrame(data, columns=columns)



num_clusters = 3  # Например, вы хотите 3 кластера
m = 2  # Параметр неопределенности


values = df.values.T

print(values)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    values, num_clusters, m=m, error=0.005, maxiter=1000, init=None
)


cluster_membership = np.argmax(u, axis=0)


df['Cluster'] = cluster_membership + 1  # +1, чтобы кластеры начинались с 1




fig, axes = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=True)

for i in range(3):
    for j in range(5):
        idx = i * 5 + j
        axes[i, j].scatter(df.iloc[:, idx], cluster_membership, c=cluster_membership, cmap='viridis', s=30)
        axes[i, j].set_title(f'Question {idx+1}')

plt.suptitle('Fuzzy C-means Clustering of Conference Interest')
plt.show()





