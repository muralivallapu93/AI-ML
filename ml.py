
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import dill
df = pd.read_csv("loan_data.csv")
# print(df.columns)
df.drop_duplicates(inplace=True)
y = df['loan_status']
df = df.drop(columns=["loan_status"], axis=1)
with open("pipeline1.pkl", 'rb') as file:
    pipe = dill.load(file)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=.20,random_state=42)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

r = accuracy_score(y_test, y_pred)
print(r)