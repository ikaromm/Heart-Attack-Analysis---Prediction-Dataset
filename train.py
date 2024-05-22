# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")

# %%

df.isnull().sum().T

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("output", axis=1), df["output"], test_size=0.2, stratify=df["output"]
)
# %%

y_test.mean(), y_train.mean()
