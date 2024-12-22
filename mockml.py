import pandas as pd 
import numpy as np 
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import (
    Pipeline, 
    make_pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    precision_score, 
    classification_report
)
from sklearn.feature_selection import f_regression, SelectKBest


df = pd.read_csv("F:\ML\Dataset.7z\Dataset\Mock_employees\Cleaned_employees_final_dataset.csv")
# print(df.isna().sum())

df.drop(columns=['employee_id'], inplace=True)
target_column = "awards_won"
df_indepent = df.drop(target_column, axis=1)
df_target = df[target_column]

cat_col = df_indepent.select_dtypes(include="object").columns.to_list()
num_col = df_indepent.select_dtypes(exclude="object").columns.tolist()
# print(df_indepent.columns.to_list())
# print(cat_col)
# print( num_col)

columntransforms = ColumnTransformer(
    transformers=[
        ("categorical_columns", OneHotEncoder(), cat_col),
        ("numerical_col", StandardScaler(), num_col)
    ],
    remainder='passthrough'
)
x_transformed = columntransforms.fit_transform(df_indepent)
selector = SelectKBest(score_func=f_regression, k=5)
x_selected = selector.fit_transform(x_transformed, df_target)

print(x_selected)
x_tain, x_test, y_train, y_test = train_test_split(x_selected, df_target, test_size=.2, random_state=42)
model = LogisticRegression(random_state=0)
model.fit(x_tain, y_train)

# import tempfile
# import joblib
# import boto3
# with tempfile.TemporaryFile as temp_file:
#     joblib.dump(model, temp_file)
#     temp_file.seek(0)
#     s3_client = boto3.client("s3", region_name = 'us-east-1')
#     s3_client.upload_fileobj(temp_file, bucket_name="/")



y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
