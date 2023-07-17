import datetime

import pandas as pd
from matplotlib import pyplot as plt


def load_dataset(dataset_name):
    df = pd.read_csv(f'./data/{dataset_name}.csv', parse_dates=['Date'])

    df = df.iloc[::-1]
    start = df['Date'].iloc[0]
    end = df['Date'].iloc[-1]
    target = df['Price'].to_numpy()

    return df, start, end, target


def display_forcast(df, display_offset, forecasts, forcast_start_date, freq, filename):
    """
    Display the forecast
    :param df: 数据集 DataFrame
    :param display_offset: 展示的数据集数据长度
    :param forecasts: 预测结果
    :param forcast_start_date: 预测结果的开始日期
    :param freq: 预测结果的频率
    :param filename: 保存的文件名
    :return:
    """
    plt.figure(figsize=(20, 15))
    x = df["Date"][-display_offset:].to_numpy()
    x = [pd.Period(p, freq=freq).to_timestamp() for p in x]
    plt.plot(x, df["Price"][-display_offset:], color="black")
    for forecast in forecasts:
        forecast.start_date = forcast_start_date
        forecast.plot()
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    plt.xticks(rotation=45, fontsize="xx-large")
    plt.yticks(fontsize="xx-large")
    plt.savefig(f"{filename}_{datetime.datetime.today().strftime('%Y%m%d')}.png")
