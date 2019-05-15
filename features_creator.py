from dateutil.relativedelta import relativedelta as rd
from libs import trading_lib as tl

import pandas as pd; pd.options.display.max_rows = 20_000


PATH_TO_SAVE = 'database/'

if __name__ == "__main__":
    df = pd.read_excel('database/DJI Features.xlsx', converters={'Date': pd.to_datetime})

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
