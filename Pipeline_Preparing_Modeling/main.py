import joblib
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def filter_cols(df):
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)

def drop_transformed_cols(df):
    columns_to_drop = [
        'year',
        'model',
        'fuel',
        'odometer',
        'title_status',
        'transmission',
        'state',
        'short_model',
        'age_category'
    ]
    return df.drop(columns_to_drop, axis=1)


def calculate_outliers(data):
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    return boundaries

def replace_outliers_year(df):
    df1 = df.copy()
    boundaries = calculate_outliers(df1['year'])

    for i in df1[df1['year'] < boundaries[0]].index:
        df1.loc[i, 'year'] = round(boundaries[0])
    for i in df1[df1['year'] > boundaries[1]].index:
        df1.loc[i, 'year'] = round(boundaries[1])
    return df1

def short_model(x):
    if not pd.isna(x):
        return x.lower().split(' ')[0]
    else:
        return x

# Добавление фичи "short_model" – это первое слово из колонки model
def mkfeature_short_model(df):
    df1 = df.copy()
    for i in df1.index:
        df1.loc[i, 'short_model'] = short_model(df1.loc[i, 'model'])
    return df1

# Добавление фичи age_category
def mkfeature_age_cat(df):
    df1 = df.copy()
    for i in df1.index:
        if df1.loc[i, 'year'] > 2013:
            df1.loc[i, 'age_category'] = 'new'
        elif df1.loc[i, 'year'] < 2006:
            df1.loc[i, 'age_category'] = 'old'
        else:
            df1.loc[i, 'age_category'] = 'average'
    return df1


def main():
    print('Price Category Prediction Pipeline')

    df = pd.read_csv('data/homework.csv')

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    numerical_transformer = Pipeline(steps=[
        ('year_outliers_replace', FunctionTransformer(replace_outliers_year)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    col_preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include='object'))
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_cols)),
        ('mkfeature_short_model', FunctionTransformer(mkfeature_short_model)),
        ('mkfeature_age_cat', FunctionTransformer(mkfeature_age_cat)),
        ('column_transform', col_preprocessor),
       # ('drop_cols', FunctionTransformer(drop_transformed_cols))
    ])

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'price_category_prediction_pipe.pkl')

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
