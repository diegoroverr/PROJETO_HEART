# ❤️ Análise de Perfis de Insuficiência Cardíaca

Projeto desenvolvido para a **Avaliação do 2º Bimestre**, com o objetivo de aplicar técnicas de *data mining* (clusterização K-Means) para identificar perfis de pacientes a partir da base de dados *Heart Failure Clinical Records*.

---

## Requisitos da Avaliação

O projeto foi estruturado para atender aos seguintes requisitos:

* Normalizar os dados.
* Determinar o número ótimo de clusters (K) e treinar o modelo.
* Descrever os clusters obtidos (centroides).
* Determinar a qual cluster um novo paciente pertence.

---

## Conjunto de Dados

Foi utilizada a base `heart_failure_clinical_records_dataset.csv`.

* **Fonte:** [Heart Failure Clinical Records Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
* **Características Principais:** O dataset inclui 13 características clínicas, como `age`, `anaemia`, `ejection_fraction`, `serum_creatinine`, `smoking`, etc.
* **Observação:** A coluna alvo `DEATH_EVENT` foi removida antes da clusterização, pois este é um exercício de aprendizado não-supervisionado.

---

## ⚙️ Metodologia e Pipeline

O projeto é executado através de um menu principal (`main.py`) que orquestra os seguintes scripts:

1.  **Pré-Processamento (`normalizar.py`):**
    * Carrega os dados brutos.
    * Separa as colunas em **contínuas** (ex: `age`, `platelets`) e **binárias** (ex: `smoking`, `diabetes`).
    * Aplica a normalização `MinMaxScaler` (escala 0-1) *apenas* nas colunas contínuas.
    * Salva o dataset processado (`dados_preprocessados_heart.csv`) e o modelo de normalização (`modelo_normalizador_heart.model`).

2.  **Treinamento do Modelo (`clusterizar.py`):**
    * Carrega os dados processados.
    * Calcula o **K ótimo** (número ideal de clusters) usando o **Método do Cotovelo (Elbow Method)**, analisando a distorção para K de 2 a 20.
    * Gera o gráfico `distorcoes_heart.jpg` para visualização do cotovelo.
    * Treina o modelo K-Means final com o K ótimo encontrado.
    * Salva o modelo de clusterização treinado (`cluster_heart.model`).

3.  **Descrição dos Clusters (`descrever_centroides.py`):**
    * Carrega os modelos (`cluster_heart.model` e `modelo_normalizador_heart.model`).
    * Extrai os centroides de cada cluster.
    * **Reverte a normalização** dos centroides para que possam ser interpretados em seus valores originais (ex: idade em anos, em vez de um valor entre 0 e 1).
    * Exibe a descrição de cada perfil de paciente médio por cluster.

4.  **Classificação de Novo Paciente (`processar_paciente_desconhecido.py`):**
    * Carrega um exemplo de "novo paciente" (definido em um dicionário Python).
    * Aplica o *mesmo* pipeline de pré-processamento (normalização) neste paciente.
    * Utiliza o modelo K-Means treinado para prever a qual cluster este novo paciente pertence.

---

## Tecnologias Utilizadas

* Python 3.x
* pandas
* scikit-learn (sklearn)
* matplotlib
* colorama
* tqdm

---

## 🚀 Como Executar

Siga os passos abaixo para rodar o projeto.

**1. Clone o repositório:**  
git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO

**2. Instale as dependências:(Recomendado usar um ambiente virtual: python -m venv venv)** 
python -m pip install pandas scikit-learn matplotlib colorama tqdm

**3. Execute o menu principal:**
python main.py

**4. Siga as instruções no menu: O programa apresentará um menu. Execute as opções na ordem (1, 2, 3, 4) para ver o fluxo completo do projeto.**

[1] Pré-Processar dados (Normalizar)
[2] Treinar Modelo (Achar K e Clusterizar)
[3] Descrever Centroides
[4] Classificar Novo Paciente
[9] (Re)Instalar Dependências
[0] Sair

**Autores (Equipe)**
- Diego Rover Rodrigues
- Daniel Taboga

