from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score

from libs import trading_lib as tl
from datetime import datetime

import pandas as pd; # pd.options.display.max_rows = 40_000; pd.set_option('display.max_columns', 40_000)
import numpy as np

""" Мысли
Бутстреп для таймсерий - это пиздец
Тайм-сплит:
1) Соседи не выше 51%
2) Регрессия 61%. C=0.0001
3) Лес 60%. n_estimators=1_000, max_depth=4
4) SVC 57%
5) SGDClassifier 60% + Nystroem
6) XGboosting 59%


"""

train_date = datetime(2019, 1, 1)
test_date = datetime(1917, 1, 1)


def features_pipeline(df) -> np.array:
    perc_features = df.loc[:, 'Momentum_perc_1':'Annual_vol_perc_12']
    onehot_features = df.loc[:, 'Momentum_bool_1':]

    return np.c_[StandardScaler().fit_transform(perc_features), onehot_features.values]


def kNN():
    # feature_map_nystroem = Nystroem(gamma=.2, random_state=17, n_components=162)
    # data_transform = feature_map_nystroem.fit_transform(X_train)

    model = LogisticRegression(C=0.0001, n_jobs=-1, random_state=17)
    ss = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in ss.split(np_X_train):
        model.fit(np_X_train[train_index], y_train[train_index])
        y_predict = model.predict(np_X_train[test_index])

        print(len(y_train[test_index][y_train[test_index]==1]) / len(y_train[test_index][y_train[test_index]==0]))
        print(accuracy_score(y_train[test_index], y_predict))

        # new_frame = pd.DataFrame({'Date': date[test_index], 'Predict': y_predict})
        # print(new_frame.loc[new_frame['Predict'] < 0.35])

    # start = 0
    # for i in range(start, np_X_train.shape[1]):
    #     X = np_X_train[:, start:i+1]
    # scores = cross_val_score(model, np_X_train, y_train, cv=5, n_jobs=-1)
    # print(f"Last feature name {features_names[i]}. "
    #       f"Features_quantity {len(features_names[start:i+1])}. "
    #       f"Score: {scores.mean()}")
    # print(f"Score: {scores.mean()}")

    # model.fit(np_X_train, y_train)
    #
    # df_test = pd.read_csv('teach_data/X_test.csv')
    # y = df_test['y_month']
    # df_test.drop(axis=1, columns=['Date', 'y_month', 'Close'], inplace=True)
    # np_X_test = features_pipeline(df_test)
    #
    # y_predict = model.predict(np_X_test)
    # print(accuracy_score(y, y_predict))


X_df = pd.read_csv('database/DJI Features_8.csv', converters={'Date': pd.to_datetime})
X_train = X_df[X_df['Date'] <= train_date]
date = X_train['Date']
y_train = X_train['y_month']
X_train.drop(axis=1, columns=['Date', 'y_month', 'Close'], inplace=True)

features_names = X_train.columns
np_X_train = features_pipeline(X_train)

# Булы от 80 до 113
print(features_names[99:113])
np_X_train = np_X_train

kNN()
# input()

# model = RandomForestClassifier(n_estimators=1_000, max_depth=10,random_state=17, n_jobs=-1)
# feat_importance = dict(zip(features_names, model.feature_importances_))
# print(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))
