# Análise de Influenciadores do Instagram com Regressão Linear

# Descrição do Projeto
Este projeto visa analisar dados de influenciadores do Instagram e aplicar técnicas de regressão linear para prever a taxa de engajamento dos perfis. O processo inclui desde a limpeza e normalização dos dados até a visualização e avaliação do modelo preditivo. 

# Instalação
1. Clone o repositório:
   git clone https://github.com/VicBia/instaDataset.git

2. Instale as bibliotecas possíveis:
pip install numpy pandas matplotlib seaborn scikit-learn

# Como Executar
1. Importe as bibliotecas:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
2. Carregue o arquivo CSV e visualize os dados:
url = '/content/top_insta_influencers_data.csv'
insta_df = pd.read_csv(url)
insta_df.head()
3. Execute o processo de limpeza, normalização e remoção de outliers conforme código fornecido.
4. Realize a análise e visualize os gráficos de dispersão.
5. Divida os dados em treino e teste e treine o modelo de regressão linear.
6. Verifique os resultados e a precisão do modelo.

# Estrutura dos Arquivos
- top_insta_influencers_data.csv: Conjunto de dados de influenciadores do Instagram.
- main_script.py: Script contendo o código principal para análise.
- README.md: Documentação do projeto.

# Tecnologias Utilizadas
Python
Bibliotecas : NumPy, pandas, matplotlib, seaborn,scikit-learn

# Autores e Colaboradores
Anna Miranda e Victoria Reis - Desenvolvimento do código e análise de dados.

