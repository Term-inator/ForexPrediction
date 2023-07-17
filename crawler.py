import os

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'https://www.investing.com/currencies/usd-cny-historical-data'


def crawler(use_cache=True):
    if not (use_cache and os.path.exists('cache.html')):
        header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36", }
        res = requests.get(url, headers=header)

        print(f'request {url}')

        with open('cache.html', 'w', encoding='utf-8') as f:
            f.write(res.text)

    with open('cache.html', 'r', encoding='utf-8') as f:
        html = f.read()

        soup = BeautifulSoup(html, 'html.parser')
        table_wrapper = soup.findAll('div', {'class': 'border border-main'})
        if len(table_wrapper) != 1:
            raise Exception('table_wrapper length error')
        history_tables = table_wrapper[0].findAll('table')

        if len(history_tables) != 1:
            raise Exception('history_tables length error')
        else:
            history_table = history_tables[0]

        history_table_head = history_table.find('thead')
        history_table_head_row = history_table_head.find('tr')
        history_table_body = history_table.find('tbody')
        history_table_rows = history_table_body.findAll('tr')

        columns = []
        for th in history_table_head_row.findAll('th'):
            columns.append(th.text)

        data = []
        for i, row in enumerate(history_table_rows):
            # 舍弃最新的数据（可能没更新完）
            if i == 0:
                continue
            row_data = row.findAll('td')

            def f(x):
                if len(x) == 0:  # 列 'Vol.' 没有值
                    return np.NAN
                else:
                    return x.text

            data.append(map(f, row_data))

        df = pd.DataFrame(data, columns=columns)
        df['Date'] = pd.to_datetime(df['Date'])
        return df


if not os.path.exists('data/USD_CNY Historical Data.csv'):
    print(f'download data from {url} first')
else:
    df = pd.read_csv('data/USD_CNY Historical Data.csv', parse_dates=['Date'])
    old_end = df['Date'].iloc[0]
    today = pd.Timestamp.today()
    # 25天内没更新
    # 周末没有数据 + 二月最短 28 天，所以设置 25 天
    if old_end < today - pd.Timedelta(days=25):
        print(f'download data from {url} first')

    new_df = crawler()
    new_end = new_df['Date'].iloc[0]  # 新数据的结束日期

    # 新数据的结束日期在昨天之前，说明数据已经更新完毕，重新爬取
    if new_end < today - pd.Timedelta(days=3):
        new_df = crawler(use_cache=False)

    # 保证数据类型一致
    for col in df.columns:
        new_df[col] = new_df[col].astype(df[col].dtype)

    index_to_remove = []  # 需要删除的行索引
    for i in range(len(new_df)):
        if new_df['Date'].iloc[i] in df['Date'].values:
            # 判断所有数据是否相等
            for col in new_df.columns:
                # 列 'Vol.' 没有值
                if col == 'Vol.':
                    continue
                old_data = df[df['Date'] == new_df['Date'].iloc[i]].iloc[0][col]

                # 去掉开头的 '+'
                if col == 'Change %':
                    if old_data[0] == '+':
                        old_data = old_data[1:]
                    if new_df.loc[i, col][0] == '+':
                        new_df.loc[i, col] = new_df.loc[i, col][1:]

                # 日期相同，其他数据不同，抛出异常
                if new_df.loc[i, col] != old_data:
                    raise Exception(f'{new_df["Date"].iloc[i]} already exists but not equal')
            print(f'{new_df["Date"].iloc[i]} already exists')
            index_to_remove.append(i)

    new_df.drop(index_to_remove, inplace=True)

    if len(new_df) > 0:
        res = pd.concat([new_df, df], ignore_index=True)
        # res['Date'] = res['Date'].strftime('%m/%d/%Y')
        # res.sort_values(by='Date', inplace=True, ascending=False)
        res.to_csv('data/USD_CNY Historical Data.csv', index=False, date_format='%m/%d/%Y')
        print(f'update data from {url}')
    else:
        print(f'no new data from {url}')
