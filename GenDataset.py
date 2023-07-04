import os

from forex_python.converter import CurrencyRates
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_dataset(load_dict):
    """
    确保每个日期都有数据
    :param load_dict:
    :return:
    """
    start = load_dict['start']
    end = load_dict['end']
    target = load_dict['target']
    freq = load_dict['freq']
    delta = datetime.timedelta(days=1)
    for t in target:
        print(f'{start.strftime("%Y/%m/%d")} {t}')
        start += delta

    if start != end + delta:
        print(f'{start.strftime("%Y/%m/%d")} {end.strftime("%Y/%m/%d")}')
        raise Exception('Dataset Error')

# start, end, target, freq
dateset_name = 'Forex'
start = datetime.datetime(2005, 4, 1)
today = datetime.datetime.today()
today = datetime.datetime(today.year, today.month, today.day)

delta = datetime.timedelta(days=1)
freq = 'D'
if not os.path.exists(f'./data/{dateset_name}.npy'):
    data = np.array([])
    begin = start
else:
    load_dict = np.load(f'./data/{dateset_name}.npy', allow_pickle=True)
    load_dict = load_dict.item()
    begin = load_dict['end'] + delta
    data = load_dict['target']
    print(f'Loaded {dateset_name} from {load_dict["start"].strftime("%Y/%m/%d")} to {load_dict["end"].strftime("%Y/%m/%d")}')
    check_dataset(load_dict)

end = today
print(f'from {begin.strftime("%Y/%m/%d")} to {end.strftime("%Y/%m/%d")}')
duration = (end - begin).days
if duration <= 0:
    print('No need to update')
    exit(0)

if not os.path.exists('./data'):
    os.mkdir('./data')

c = CurrencyRates()

tmp_date = begin
tmp_list = []

with tqdm(total=duration) as pbar:
    for i in range(duration):
        tmp_list.append(c.get_rate('USD', 'CNY', date_obj=tmp_date))
        if i % 100 == 0:
            data = np.concatenate((data, np.array(tmp_list)))
            tmp_list = []
            np.save(f'./data/{dateset_name}.npy', {'start': start, 'end': tmp_date, 'target': data, 'freq': freq})
        tmp_date += delta
        pbar.set_description(f'Processing {tmp_date}')
        pbar.update(1)
data = np.concatenate((data, np.array(tmp_list)))
print(data)


np.save(f'./data/{dateset_name}.npy', {'start': start, 'end': end, 'target': data, 'freq': freq})
