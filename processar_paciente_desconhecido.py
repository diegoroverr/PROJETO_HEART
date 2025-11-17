# processar_paciente_desconhecido.py

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
colunas_todas_ordenadas = colunas_continuas + colunas_binarias

# --- 3. Entrar dados de um novo paciente ---
# (Este é um paciente fictício para exemplo)
paciente = {
    'age': 60.0,
    'creatinine_phosphokinase': 750,
    'ejection_fraction': 35,
    'platelets': 250000.0,
    'serum_creatinine': 1.8,
    'serum_sodium': 135,
    'time': 100,
    
    'anaemia': 1, # 1 = Sim
    'diabetes': 0, # 0 = Não
    'high_blood_pressure': 1, # 1 = Sim
    'sex': 1, # 1 = Homem
    'smoking': 0 # 0 = Não
}
print(f"Dados do novo paciente (antes de processar):\n{paciente}\n")

# Converter paciente para DataFrame
df_paciente = pd.DataFrame(paciente, index=[0])

# --- 4. Processar o paciente (exatamente como no treino) ---

# Separar contínuas e binárias
paciente_continuas = df_paciente[colunas_continuas]
paciente_binarias = df_paciente[colunas_binarias]

# Aplicar normalização (APENAS .transform(), NUNCA .fit())
paciente_continuas_norm_array = normalizador_num.transform(paciente_continuas)

# Converter de volta para DataFrame
paciente_continuas_norm = pd.DataFrame(
    paciente_continuas_norm_array, 
    columns=colunas_continuas
)

# Juntar contínuas (normalizadas) e binárias (originais)
df_paciente_processado = pd.concat(
    [paciente_continuas_norm.reset_index(drop=True), 
     paciente_binarias.reset_index(drop=True)], 
    axis=1
)

# Garantir a ordem exata das colunas
df_paciente_final = df_paciente_processado[colunas_todas_ordenadas]

# --- 5. Fazer a predição ---
cluster_predito = clusters.predict(df_paciente_final)[0]

print("=" * 40)
print(f"  O novo paciente pertence ao CLUSTER: {cluster_predito}")
print("=" * 40)
print("\n(Use 'descrever_centroides.py' para entender o que este cluster significa)")