from dateutil.relativedelta import relativedelta as rd
from libs import trading_lib as tl

import pandas as pd; pd.options.display.max_rows = 20_000


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


if __name__ == "__main__":

    df = pd.read_csv('database/DJI Features_1.csv', converters={'Date': pd.to_datetime})

    # Create momentums
    for i in range(1, 25):
        df[f"Momentum_perc_{i}"] = [0] * len(df)

    for day in range(0, len(df.Date)):
        print(df.Date[day])
        for i in range(1, 25):
            closes_next_month = df[df.Date[day] - rd(months=i) >= df.Date]['Close']
            if len(closes_next_month) == 0:
                continue

            df.loc[day, f"Momentum_perc_{i}"] = (closes_next_month.iloc[0] / df.Close[day] - 1) * 100

    tl.save_csv(PATH_TO_SAVE, 'DJI Features_1', df)
