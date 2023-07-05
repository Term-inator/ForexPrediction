import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model import Predictor
from tqdm import tqdm

# Load data from a CSV file into a PandasDataset
dataset_name = 'USD_CNY Historical Data'
df = pd.read_csv(f'./data/{dataset_name}.csv', parse_dates=['Date'])

df = df.iloc[::-1]
end = df['Date'].iloc[-1]
start = df['Date'].iloc[0]
target = df['Price'].to_numpy()
print(target.shape)

prediction_length = 7
freq = "D"

for i in range(prediction_length):
    target = np.append(target, 7)
print(target.shape)
target = target.reshape(1, -1)

end = pd.Period(end, freq=freq)

# Load the trained model
predictor_path = "./model/"
predictor = Predictor.deserialize(Path(predictor_path))

future_list = [{"target": x, "start": start} for x in target]
future_ds = ListDataset(
    future_list, freq=freq
)

# forecasts = list(model.predict(test_ds))
forecast_it, ts_it = make_evaluation_predictions(
    dataset=future_ds,
    predictor=predictor,
    num_samples=100
)
forecasts = list(tqdm(forecast_it, total=len(future_ds)))

# Plot the predictions
display_offset = 30
x = df["Date"][-display_offset:].to_numpy()
x = [pd.Period(p, freq=freq).to_timestamp() for p in x]
plt.plot(x, df["Price"][-display_offset:], color="black")
for forecast in forecasts:
    forecast.start_date = end
    forecast.plot()
plt.legend(["True values"], loc="upper left", fontsize="xx-large")
plt.savefig(f"forecast_{datetime.datetime.today().strftime('%Y%m%d')}.png")