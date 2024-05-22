# %%
import pandas as pd
from func import MemorySaver
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn import pipeline
from sklearn import ensemble
from sklearn import model_selection

# Load the dataset
df = pd.read_csv("heart.csv")
df.head()

# %%
df.isnull().sum().T

# %%
df.dtypes

# %%
# Instantiate the MemorySaver class
memory_saver = MemorySaver()

# Use the instance to call reduce_mem_usage method
df = memory_saver.reduce_mem_usage(df)
df.dtypes

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("output", axis=1), df["output"], test_size=0.2, stratify=df["output"], random_state=42
)

# %%
y_test.mean(), y_train.mean()

# %%
model = ensemble.RandomForestClassifier(random_state=42)

params = {
    "n_estimators": [100,150,200,250],
    "min_samples_leaf": [10,20,25,30,50],
    "max_depth": [3,4,5,7,9],
    "max_features": [0.3,0.5,0.7],
    "criterion": ["entropy"],
    "bootstrap": [True],
}

grid = model_selection.GridSearchCV(model,
                                    param_grid=params,
                                    n_jobs=-1,
                                    scoring='roc_auc')


meu_pipeline = pipeline.Pipeline([
     ('model', grid),    
    ])

meu_pipeline.fit(X_train, y_train)
# %%

first = pd.DataFrame(grid.cv_results_)
first.to_csv('first.csv')

# %%
first.query("rank_test_score<=5")
# %%
y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)
# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print("Acurárica base train:", acc_train)
print("Acurárica base test:", acc_test)

auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba[:,1])
print("\nAUC base train:", auc_train)
print("AUC base test:", auc_test)

# %%

features = X_train.columns

# %%
f_importance = meu_pipeline[-1].best_estimator_.feature_importances_
(pd.Series(f_importance, index=features).sort_values(ascending=False))


# %%

metrics.confusion_matrix(y_test, y_test_predict)

metrics.recall_score(y_test, y_test_predict)

# %%
