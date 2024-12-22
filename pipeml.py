import pandas as pd 
import numpy as  np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("F:\ML\Dataset.7z\Dataset\Mock_employees\Cleaned_employees_final_dataset.csv")
# print(df.isna().sum())

df.drop(columns=['employee_id'], inplace=True)
target_column = "awards_won"
df_indepent = df.drop(target_column, axis=1)
df_target = df[target_column]

cat_col = df_indepent.select_dtypes(include="object").columns.to_list()
num_col = df_indepent.select_dtypes(exclude="object").columns.tolist()


cat_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", OneHotEncoder())
    ]
)
num_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler1", StandardScaler())
    ]
)
processor = ColumnTransformer(
    transformers=[
        ("cat_pipe", cat_pipe, cat_col),
        ("num_pip", num_pipe, num_col)
    ],
    remainder='passthrough'
)
model = make_pipeline(
    processor,
    LogisticRegression()
)
X_train, X_test,y_train, y_test = train_test_split(df_indepent, df_target, test_size=.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))