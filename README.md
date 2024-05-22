# Heart-Attack-Analysis---Prediction-Dataset

## Visão Geral do Projeto

Este projeto utiliza aprendizado de máquina para prever a presença de doenças cardíacas com base em um conjunto de dados de pacientes. Para isso, foram empregadas técnicas avançadas de machine learning, como Random Forest, GridSearchCV, Pipeline e Ensemble.

## Estrutura do Projeto

### 1. Carregamento e Preparação dos Dados

Primeiramente, carregamos o conjunto de dados `heart.csv` e fizemos a inspeção inicial dos dados:

```
import pandas as pd

# Carregar o dataset
df = pd.read_csv("heart.csv")
df.head()

# Verificar valores nulos
df.isnull().sum().T

# Verificar tipos de dados
df.dtypes

```

### 2. Redução do Uso de Memória

Utilizamos a classe `MemorySaver` para otimizar o uso de memória do DataFrame:

```
from func import MemorySaver

# Instanciar a classe MemorySaver
memory_saver = MemorySaver()

# Reduzir o uso de memória
df = memory_saver.reduce_mem_usage(df)
df.dtypes

```

### 3. Divisão dos Dados

Dividimos os dados em conjuntos de treino e teste, mantendo a proporção das classes:

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("output", axis=1), df["output"], test_size=0.2, stratify=df["output"], random_state=42
)

```

### 4. Construção e Treinamento do Modelo

Utilizamos um Random Forest Classifier com GridSearchCV para encontrar os melhores hiperparâmetros. Além disso, utilizamos um Pipeline para facilitar o processo de treino:

```
from sklearn import ensemble, model_selection, pipeline

# Definir o modelo
model = ensemble.RandomForestClassifier(random_state=42)

# Definir os parâmetros para o GridSearchCV
params = {
    "n_estimators": [100, 150, 200, 250],
    "min_samples_leaf": [10, 20, 25, 30, 50],
    "max_depth": [3, 4, 5, 7, 9],
    "max_features": [0.3, 0.5, 0.7],
    "criterion": ["entropy"],
    "bootstrap": [True],
}

# Configurar o GridSearchCV
grid = model_selection.GridSearchCV(model, param_grid=params, n_jobs=-1, scoring='roc_auc')

# Definir o Pipeline
meu_pipeline = pipeline.Pipeline([
    ('model', grid),
])

# Treinar o modelo
meu_pipeline.fit(X_train, y_train)
```

### 5. Avaliação do Modelo

Após o treinamento, avaliamos o modelo usando várias métricas de desempenho:


```
from sklearn import metrics

# Previsões no conjunto de treino e teste
y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:, 1]
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)

# Avaliação de acurácia
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print("Acurácia base train:", acc_train)
print("Acurácia base test:", acc_test)

# Avaliação de AUC
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba[:, 1])
print("\nAUC base train:", auc_train)
print("AUC base test:", auc_test)

```


### 6. Importância das Features

Analisamos a importância das features para entender quais variáveis mais impactam o modelo:

```
features = X_train.columns
f_importance = meu_pipeline[-1].best_estimator_.feature_importances_
(pd.Series(f_importance, index=features).sort_values(ascending=False))

```


### 7. Matrizes de Confusão e Recall

Por fim, geramos a matriz de confusão e calculamos a taxa de recall:

```
# Matriz de confusão
metrics.confusion_matrix(y_test, y_test_predict)

# Taxa de recall
metrics.recall_score(y_test, y_test_predict)

```


## Ferramentas e Bibliotecas Utilizadas

* **Pandas** : Para manipulação e análise dos dados.
* **Scikit-learn** : Para modelagem, incluindo Random Forest, GridSearchCV, Pipeline e métricas de avaliação.
* **MemorySaver** : Classe personalizada para otimização de uso de memória.

Este projeto demonstra a aplicação prática de técnicas de machine learning para resolver problemas do mundo real, como a previsão de doenças cardíacas, utilizando uma abordagem sistemática e eficiente. Através do uso de pipelines e tuning de hiperparâmetros com GridSearchCV, conseguimos otimizar o desempenho do modelo e garantir que ele seja robusto e generalizável.

### Próximos Passos

Para melhorar ainda mais o modelo, podemos considerar as seguintes ações:

* **Experimentação com Outras Técnicas de Ensemble** : Testar algoritmos como Gradient Boosting e XGBoost.
* **Feature Engineering** : Criar novas features ou transformar as existentes para melhor capturar a variabilidade dos dados.
* **Validação Cruzada** : Implementar validação cruzada k-fold para uma avaliação mais robusta do modelo.
* **Implementação de Métricas Adicionais** : Avaliar o modelo com métricas adicionais como F1-score, precisão e especificidade para obter uma visão mais completa de seu desempenho.

Com essas melhorias, podemos aumentar a precisão e a utilidade prática do modelo.
