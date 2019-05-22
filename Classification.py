from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from libs import trading_lib as tl

import pandas as pd; pd.options.display.max_rows = 40_000; pd.set_option('display.max_columns', 40_000)
import numpy as np


def features_pipeline():
    pass


# df = pd.read_csv('database/DJI Features_8.csv', converters={'Date': pd.to_datetime})

dates = df['Date']
y = df['y_month']
df.drop(axis=1, columns=['y_month', 'Close', 'Date'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, df['y_month'], test_size=.1, random_state=17)

features_names = X_train.columns
train_perc_features = X_train.loc[:, 'Momentum_perc_1':'Annual_vol_perc_12']
train_onehot_features = X_train.loc[:, 'Momentum_bool_1':]
test_perc_features = X_test.loc[:, 'Momentum_perc_1':'Annual_vol_perc_12']
test_onehot_features = X_test.loc[:, 'Momentum_bool_1':]


train_perc_features = StandardScaler().fit_transform(train_perc_features)
X_train = np.c_[train_perc_features, train_onehot_features.values]
test_perc_features = StandardScaler().fit_transform(test_perc_features)
X_test = np.c_[test_perc_features, test_onehot_features.values]

logit = RandomForestClassifier(n_estimators=1_000, max_depth=10,random_state=17, n_jobs=-1)
logit.fit(X_train, y_train)

y_predict = logit.predict(X_test)
print(accuracy_score(y_test, y_predict))

feat_importance = dict(zip(features_names, logit.feature_importances_))
print(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))
