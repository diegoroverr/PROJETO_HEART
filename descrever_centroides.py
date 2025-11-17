# descrever_centroides.py

from pickle import load
import pandas as pd
import numpy as np

try:
    # --- 1. Carregar modelos ---
    clusters = load(open('cluster_heart.model', 'rb'))
    normalizador_num = load(open('modelo_normalizador_heart.model', 'rb'))
except FileNotFoundError:
    print("Erro: Modelos ('cluster_heart.model' ou 'modelo_normalizador_heart.model') não encontrados.")
    print("Por favor, execute 'normalizar.py' e 'clusterizar.py' primeiro.")
    exit()

# --- 2. Colunas (mesma ordem do treino) ---
colunas_continuas = [
    'age', 
    'creatinine_phosphokinase', 
    'ejection_fraction', 
    'platelets', 
    'serum_creatinine', 
    'serum_sodium', 
    'time'
]
colunas_binarias = [
    'anaemia', 
    'diabetes', 
    'high_blood_pressure', 
    'sex', 
    'smoking'
]
colunas_todas = colunas_continuas + colunas_binarias

# --- 3. Obter centroides ---
centroides = clusters.cluster_centers_
df_centroides = pd.DataFrame(centroides, columns=colunas_todas)

# --- 4. Reverter normalização (Apenas contínuas) ---
df_centroides[colunas_continuas] = normalizador_num.inverse_transform(df_centroides[colunas_continuas])

# --- 5. Impressão / descrição dos centroides ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n===== CENTROIDES (reconstruídos) =====\n")
# Arredondar para facilitar leitura
print(df_centroides.round(2))

print("\n===== DESCRIÇÃO LEGÍVEL POR CLUSTER =====\n")
for i, row in df_centroides.iterrows():
    print(f"--- Cluster {i} ---")
    
    print("  Características Contínuas (Médias):")
    for col in colunas_continuas:
        print(f"    - {col}: {row[col]:.2f}")
        
    print("\n  Características Binárias (Proporção 0-1):")
    for col in colunas_binarias:
        # Um valor de 0.80 significa 80% de '1' (Sim)
        print(f"    - {col}: {row[col]:.2f} ({(row[col]*100):.0f}% 'Sim')")
    print("-" * (20 + len(str(i))))