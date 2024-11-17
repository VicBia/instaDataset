# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
sns.set(style='whitegrid')

# Importando df
url = '/content/top_insta_influencers_data.csv'
insta_df = pd.read_csv(url)
insta_df.head()

# Transformando valores de string como 2M em numéricos como 2000000
replace = {'b': 'e9', 'm': 'e6', 'k': 'e3', '%': ''}
convert_column = ['total_likes', 'posts', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like']
insta_df[convert_column] = insta_df[convert_column].replace(replace, regex=True).astype(float)
insta_df[convert_column]

# Visualizando as estatísticas descritivas dos dados (outliers percebidos)
insta_df.describe()

# Removendo outliers
def remove_outliers_iqr(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Colunas para remover outliers
columns_to_remove_outliers = ['total_likes', 'posts', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like']

# Removendo outliers
insta_df_no_outliers = remove_outliers_iqr(insta_df.copy(), columns_to_remove_outliers)

# DataFrame sem outliers
insta_df = insta_df_no_outliers
insta_df.describe()

# Normalização colunas
def normalize_columns(df, columns):
    for column in columns:
        # Calculando o minimo e maximo valor da coluna
        min_val = df[column].min()
        max_val = df[column].max()
        # Aplicando escala Min-Max na coluna
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

columns_to_normalize = ['total_likes', 'posts', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like']

insta_df_normalized = normalize_columns(insta_df.copy(), columns_to_normalize)

#  DataFrame normalizado
insta_df_normalized.head()

# Visualizando as estatísticas descritivas dos dados
insta_df_normalized.describe()

def print_corr_scores(df, column_name):
  # Calcular correlações
  corr_scores = df.corr()[column_name]

  #  Scores da correlação
  for col, score in corr_scores.items():
    if col != column_name:
      print(f"Correlation between {column_name} and {col}: {score:.2f}")

insta_df = insta_df_normalized[['influence_score', 'posts', 'followers', 'avg_likes', 'new_post_avg_like', 'total_likes', '60_day_eng_rate']]
print_corr_scores(insta_df, '60_day_eng_rate')

insta_num = insta_df.select_dtypes(include=[np.number])

# Verificando a correlação entre as variáveis
corr_matrix = insta_num.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title('Correlação entre as variáveis')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

# Definir as colunas para analisar
columns_to_plot = ['followers', 'avg_likes', 'new_post_avg_like', 'total_likes', 'posts', 'influence_score', '60_day_eng_rate']

# Selecionar  essas colunas e remover linhas com valores ausentes
insta_df_selected = insta_df[columns_to_plot].dropna()

# Criar scatter plots para cada uma das características selecionadas em relação à taxa de engajamento
plt.figure(figsize=(15, 10))
for i, column in enumerate(insta_df_selected.columns[:-1], 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=insta_df_selected, x=column, y='60_day_eng_rate')
    plt.title(f'Relação entre {column} e 60_day_eng_rate')
    plt.xlabel(column)
    plt.ylabel('60_day_eng_rate')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='new_post_avg_like', y='60_day_eng_rate', data=insta_df)
plt.title('Relacionamento entre seguidores e taxa de engajamento de 60 dias')
plt.xlabel('Média de likes por novo post')
plt.ylabel('Taxa de engajamento de 60 dias')
plt.show()

# Definindo as variáveis independentes (excluindo a variável '60_day_eng_rate' que será a dependente)
X = insta_df[['followers', 'avg_likes', 'new_post_avg_like', 'total_likes', 'posts']]
y = insta_df['60_day_eng_rate']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o objeto de regressão linear
model = LinearRegression()
# Treinando o modelo
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualizar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Real')
plt.ylabel('Previsão')
plt.title('Real vs Previsão')
plt.show()

print(f'RMSE: {mse ** 0.5:.2f}')
print(f'R²: {r2:.2f}')

# DataFrame com recursos de teste e previsões
lr_preds = model.predict(X_test)

predictions_df = X_test.copy()
predictions_df['Actual Engagement Rate'] = y_test.values
predictions_df['Predicted Engagement Rate'] = lr_preds

predictions_df.head()

