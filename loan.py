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

num_col = ["person_age", "person_income", "person_emp_exp",'loan_amnt', 'loan_intent',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score']


cat_cols = [var for var in df.columns if df[var].dtypes == 'object']
num_col1 = df.select_dtypes(exclude=['object']).columns.to_list()
print(num_col1)

num_pipline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)
cat_pipline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", OneHotEncoder())
        # ("scaler", StandardScaler())
    ]
)

pipline_processing = ColumnTransformer(
    transformers=[
    ("cat_col1", cat_pipline, cat_cols),
    ("num_pipline", num_pipline, num_col1)
]
)
pipe = make_pipeline(pipline_processing, LogisticRegression())

with open("pipeline1.pkl", 'wb') as file:
    dill.dump(pipe, file)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=.20,random_state=42)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

r = accuracy_score(y_test, y_pred)
print(r)

