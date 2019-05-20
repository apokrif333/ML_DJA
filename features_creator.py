from dateutil.relativedelta import relativedelta as rd
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


if __name__ == "__main__":

    df = pd.read_csv('database/DJI Features_2.csv', converters={'Date': pd.to_datetime})

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


