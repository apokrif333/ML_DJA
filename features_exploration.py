import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
from plotly import graph_objs as go

import pandas as pd
import numpy as np


df = pd.read_csv('database/DJI Features_8.csv', converters={'Date': pd.to_datetime})
print(f"Class_1 to Class_0: {len(df[df.y_month == 1]) / len(df[df.y_month == 0])}")

# Pandas bars
# df.groupby(['y_month', 'SMA_bool_75']).size().plot(kind='bar')
# plt.show()

# Feature probability to vector
# start_bool_idx = list(df.columns).index('Momentum_bool_1')
#
# for column in df.columns[3:start_bool_idx]:
#     group_tab = df.groupby(['y_month']).agg({column: [np.mean, np.std]})
#     print(f"{column} mean difference: {group_tab[column]['mean'][1] / group_tab[column]['mean'][0]}")
#
# for column in df.columns[start_bool_idx:]:
#     group_tab = df.groupby(['y_month']).agg({column: [np.sum]})
#     print(f"{column} base ratio diff: {group_tab[column]['sum'][1] / group_tab[column]['sum'][0]}")

# Crosstab
# cross_tab = pd.crosstab(df.y_month, df.Date.dt.day)
# print(cross_tab.loc[1, :] / cross_tab.loc[0, :])

# Plotly-boxes
# data = []
# for column in df.loc[:, 'SMA_perc_10':'SMA_perc_400'].columns:
#     for i in range(2):
#         data.append(go.Box(name=column + f"_v_{i}", y=df[df.y_month == i][column]))
# plot(data, show_link=False, filename='test_plot' + '.html')

# Corr-matrix
# new_df = df.drop(['Date', 'Close', 'y_month'], axis=1)
# # new_df = new_df.loc[:, 'Annual_vol_perc_0.5':'Annual_vol_perc_12']
# # new_df['y_month'] = df['y_month']
# # corr_matrix = new_df.corr()
# # sns.heatmap(corr_matrix)
# # plt.show()

# Hist
# sns.distplot(df['Momentum_perc_24'])
# plt.show()
# Построй гистограмму, как менялось feature probability по годам, ну или десятилеткам

# Joinplot
sns.jointplot(df['y_month'], df['Momentum_perc_2'])
plt.show()
