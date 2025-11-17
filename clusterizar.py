# clusterizar.py

from sklearn import preprocessing
from pickle import dump 
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

print("Iniciando clusterização (Cálculo do K ótimo)...")

# --- 1. Carregar dados preprocessados ---
try:
    dados = pd.read_csv('dados_preprocessados_heart.csv', sep=',')
except FileNotFoundError:
    print("Erro: Arquivo 'dados_preprocessados_heart.csv' não encontrado.")
    print("Por favor, execute 'normalizar.py' primeiro.")
    exit()

# --- 2. Calcular Distorções (Método do Cotovelo) ---
distorcoes = []
K = range(2, 21) # Intervalo reduzido para 2 a 20 (base é menor)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados,
                      cluster_model.cluster_centers_,
                      'euclidean'), axis=1) / dados.shape[0]
        )
    )

# --- 3. Gerar o gráfico das distorções ---
fig, ax = plt.subplots()
ax.plot(K, distorcoes)
ax.set(xlabel='Nº de Clusters (K)', ylabel='Distorção Média')
ax.grid()
plt.savefig('distorcoes_heart.jpg') # Salvar o gráfico
print(f"Gráfico do cotovelo salvo em 'distorcoes_heart.jpg'")

# --- 4. Determinar o número ideal de clusters (Elbow) ---
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1] 
yn = distorcoes[-1] 
distancias = [] 
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs(
        (yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0
    )
    denominador = math.sqrt(
        (yn - y0)**2 + (xn - x0)**2
    )
    distancias.append(numerador / denominador)

# K ótimo é o que tem a maior distância da reta
k_otimo = K[np.argmax(distancias)]
print(f"Número ideal de clusters (K ótimo) determinado: {k_otimo}")

# --- 5. Treinar e Salvar o Modelo Final ---
print(f"Treinando modelo final com K={k_otimo}...")
cluster_model_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10).fit(dados)

# Salvar o modelo
dump(cluster_model_final, open('cluster_heart.model', 'wb'))

print(f"Modelo de clusterização salvo em 'cluster_heart.model'")
print("Clusterização concluída.")