# normalizar.py

from sklearn import preprocessing
from pickle import dump
import pandas as pd

print("Iniciando pré-processamento...")

# --- 1. Carregar Dados ---
try:
    dados = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',')
except FileNotFoundError:
    print("Erro: Arquivo 'heart_failure_clinical_records_dataset.csv' não encontrado.")
    exit()

# --- 2. Separar Features e Target ---
# Para clusterização, removemos o evento alvo (DEATH_EVENT)
if 'DEATH_EVENT' in dados.columns:
    dados = dados.drop(columns=['DEATH_EVENT'])
else:
    print("Aviso: Coluna 'DEATH_EVENT' não encontrada.")

# --- 3. Definir Tipos de Coluna ---
# Colunas contínuas (que precisam de scaling)
colunas_continuas = [
    'age', 
    'creatinine_phosphokinase', 
    'ejection_fraction', 
    'platelets', 
    'serum_creatinine', 
    'serum_sodium', 
    'time'
]

# Colunas binárias (que já são 0 ou 1 e não precisam de get_dummies)
colunas_binarias = [
    'anaemia', 
    'diabetes', 
    'high_blood_pressure', 
    'sex', 
    'smoking'
]

# Verificar se todas as colunas estão presentes
missing_cols = [c for c in colunas_continuas + colunas_binarias if c not in dados.columns]
if missing_cols:
    print(f"Erro: Colunas faltando no CSV: {missing_cols}")
    exit()

# Separar os dataframes
dados_continuos = dados[colunas_continuas]
dados_binarios = dados[colunas_binarias]

# --- 4. Normalização (Apenas Contínuas) ---
# Construir o objeto normalizador numérico (MinMaxScaler)
normalizador_numerico = preprocessing.MinMaxScaler()

# Treinar o modelo normalizador
modelo_normalizador = normalizador_numerico.fit(dados_continuos)

# Salvar o modelo normalizador
dump(modelo_normalizador, open('modelo_normalizador_heart.model', 'wb'))

# Aplicar a normalização
dados_continuos_normalizados_array = modelo_normalizador.transform(dados_continuos)

# Converter de volta para DataFrame (preservando nomes das colunas)
dados_continuos_normalizados = pd.DataFrame(
    dados_continuos_normalizados_array, 
    columns=colunas_continuas
)

# --- 5. Juntar Dados e Salvar ---
# Resetar o índice para garantir a concatenação correta
dados_continuos_normalizados = dados_continuos_normalizados.reset_index(drop=True)
dados_binarios = dados_binarios.reset_index(drop=True)

# Juntar os dados normalizados (contínuos) com os binários (originais)
dados_finais_preprocessados = pd.concat(
    [dados_continuos_normalizados, dados_binarios], 
    axis=1
)

# Salvar o arquivo processado
dados_finais_preprocessados.to_csv('dados_preprocessados_heart.csv', sep=',', index=False)

print("Pré-processamento concluído.")
print(f"Dados salvos em 'dados_preprocessados_heart.csv'")
print(f"Modelo de normalização salvo em 'modelo_normalizador_heart.model'")