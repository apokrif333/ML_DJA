from dateutil.relativedelta import relativedelta as rd
import datetime as dt
from libs import trading_lib as tl

import pandas as pd; pd.options.display.max_rows = 20_000
import numpy as np


PATH_TO_SAVE = 'database/'


def create_y(df: pd.DataFrame):
    df['y_month'] = [0] * len(df)
    for day in range(0, len(df.Date)):
        print(df.Date[day])
        closes_next_month = df[df.Date[day] + rd(months=1) <= df.Date]['Close']

        if len(closes_next_month) != 0:
            df.loc[day, 'y_month'] = 1 if closes_next_month.iloc[0] / df.Close[day] >= 1 else 0
        else:
            print(closes_next_month)
            df.loc[day, 'y_month'] = None

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_1', df)


def create_momentums(df: pd.DataFrame):
    for i in range(1, 25):
        df[f"Momentum_perc_{i}"] = [0] * len(df)

    for day in range(0, len(df.Date)):
        print(df.Date[day])
        for i in range(1, 25):
            close_past_month = df[df.Date[day] - rd(months=i) <= df.Date]['Close'].iloc[0]
            if close_past_month == df.Close[0]:
                continue

            df.loc[day, f"Momentum_perc_{i}"] = (df.Close[day] / close_past_month - 1) * 100

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_1', df)


def create_sma(df: pd.DataFrame):
    sma_columns = [10, 20] + list(range(25, 425, 25))
    for number in sma_columns:
        np_sma = np.array(round(df['Close'].rolling(number).mean(), 2))
        df[f"SMA_perc_{number}"] = (np.array(df.Close) / np_sma - 1) * 100

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_2', df)


def create_vol(df: pd.DataFrame):
    vol_columns = [.5, .75] + list(range(1, 13))
    for i in vol_columns:
        df[f"Annual_vol_perc_{i}"] = [0] * len(df)

    for day in range(0, len(df.Date)):
        print(df.Date[day])
        for number in vol_columns:
            if number == .5:
                closes_past = df[df.Date[day] - rd(weeks=2) <= df.Date]['Close']
            elif number == .75:
                closes_past = df[df.Date[day] - rd(weeks=3) <= df.Date]['Close']
            else:
                closes_past = df[df.Date[day] - rd(months=number) <= df.Date]['Close']

            if closes_past.iloc[0] == df.Close[0]:
                continue

            prices = np.array(
                closes_past[list(range(closes_past.index[0], day + 1))]
            )
            prices_cng = np.diff(prices) / prices[:-1] * 100
            df.loc[day, f"Annual_vol_perc_{number}"] = np.std(prices_cng, ddof=1) * np.sqrt(252)

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_3', df)


def mom_sma_vol_one_hot_encoding(df: pd.DataFrame):
    for i in range(1, 25):
        df[f"Momentum_bool_{i}"] = np.array(df[f"Momentum_perc_{i}"] >= 0).astype('int')

    sma_columns = [10, 20] + list(range(25, 425, 25))
    for number in sma_columns:
        df[f"SMA_bool_{number}"] = np.array(df[f"SMA_perc_{number}"] >= 0).astype('int')

    vol_list = list(range(3, 32, 2))
    for i in range(len(vol_list)):
        if i == 0:
            df[f"Vol_bool_{vol_list[i]}"] = np.array(df['Annual_vol_perc_1'] < vol_list[i]).astype('int')
        elif i + 1 != len(vol_list):
            check_vols = np.array(vol_list[i] <= df['Annual_vol_perc_1']) == \
                         np.array(df['Annual_vol_perc_1'] < vol_list[i + 1])
            df[f"Vol_bool_{vol_list[i]}_{vol_list[i + 1]}"] = check_vols.astype('int')
        else:
            df[f"Vol_bool_{vol_list[i]}"] = np.array(vol_list[i] <= df['Annual_vol_perc_1']).astype('int')

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_4', df)


def add_time_as_feature(df: pd.DataFrame):
    # Delete sunday and saturday
    week_days = df.Date.dt.weekday
    df.drop((week_days[week_days == 6]).index, inplace=True)

    week_days = df.Date[df.Date >= dt.datetime(1953, 1, 1)].dt.weekday
    df.drop((week_days[week_days == 5]).index, inplace=True)

    # Create one-hot columns with time
    uniq_months = sorted(df.Date.dt.month.unique())
    uniq_week_days = sorted(df.Date.dt.weekday.unique())
    uniq_days = sorted(df.Date.dt.day.unique())

    for month in uniq_months:
        df[f"Month_{month}"] = (df.Date.dt.month == month).astype('int')
    for week_d in uniq_week_days:
        df[f"Week_d_{week_d}"] = (df.Date.dt.weekday == week_d).astype('int')
    for day in uniq_days:
        df[f"Day_{day}"] = (df.Date.dt.day == day).astype('int')

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_5', df)


def cut_data(df: pd.DataFrame):
    start_index = df[df['Momentum_perc_24'] != 0].index[0]
    end_index = df['y_month'][np.isnan(df['y_month'])].index[0]

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_6', df[start_index:end_index])


def check_data_wholeness(df: pd.DataFrame):
    calc_dates = np.ediff1d(np.array(df.Date))
    dates_diff = (calc_dates / 86_400 / 1_000_000_000).astype('int')
    dates_diff = np.insert(dates_diff, 0, 0)
    df['Days_jump'] = dates_diff
    print(df[df['Days_jump'] > 4])


if __name__ == "__main__":
    # !We have saturdays from start to 1952 year!
    # !We have empty data from 7/30/1914 to 12/12/1914! Для цельности обучения лучше вырезать период (12/12/1914 + 2 года)

    # 1) Добавить недельную волу. 2) Энкодинг волы завязать не на месяц, а на какой-нибудь период недель.

    df = pd.read_csv('database/DJI Features_7.csv', converters={'Date': pd.to_datetime})
