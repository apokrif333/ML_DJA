from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from libs import trading_lib as tl
from datetime import datetime

import pandas as pd; # pd.options.display.max_rows = 40_000; pd.set_option('display.max_columns', 40_000)
import numpy as np

""" Мысли
1) На лесе были опробованы тестовые сайзы [.1, .2, .3, .4], при 4-ёх разных рандом-стейтах. Среднее аккураси >0.75. Всё с бутстрепом.
2) Соседи дали >.90, при использовании более 10 фичей. На разных тестовых сайзах и 91% на основном бутстреп-тестовом файле. 


"""


def features_pipeline(df) -> np.array:
    perc_features = df.loc[:, 'Momentum_perc_1':'Annual_vol_perc_12']
    onehot_features = df.loc[:, 'Momentum_bool_1':]

    return np.c_[StandardScaler().fit_transform(perc_features), onehot_features.values]


def kNN():
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    # start = 24
    # for i in range(start, np_X_train.shape[1]):
    #     X = np_X_train[:, start:i+1]
    #     scores = cross_val_score(model, X, y_train, cv=5, n_jobs=-1)
    #     print(f"Last feature name {features_names[i-1]}. "
    #           f"Features_quantity {len(features_names[start:i+1])}. "
    #           f"Score: {scores.mean()}")

    model.fit(np_X_train, y_train)

    df_test = pd.read_csv('teach_data/X_test.csv')
    y = df_test['y_month']
    df_test.drop(axis=1, columns=['Date', 'y_month', 'Close'], inplace=True)
    np_X_test = features_pipeline(df_test)

    y_predict = model.predict(np_X_test)
    print(accuracy_score(y, y_predict))


X_df = pd.read_csv('database/DJI Features_8.csv', converters={'Date': pd.to_datetime})
y = X_df['y_month']
X_df.drop(axis=1, columns=['y_month', 'Close'], inplace=True)


X_train, X_valid, y_train, y_valid = train_test_split(X_df, y, test_size=0.15, random_state=17)
train_dates = X_train['Date']
valid_dates = X_valid['Date']
X_train.drop(axis=1, columns=['Date'], inplace=True)
X_valid.drop(axis=1, columns=['Date'], inplace=True)

features_names = X_train.columns
np_X_train = features_pipeline(X_train)
np_X_valid = features_pipeline(X_valid)

# model = RandomForestClassifier(n_estimators=1_000, max_depth=10,random_state=17, n_jobs=-1)
kNN()
input()

feat_importance = dict(zip(features_names, model.feature_importances_))
print(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))
