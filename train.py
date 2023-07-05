"""
for data from csv
"""
import os.path
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.torch import TemporalFusionTransformerEstimator
from tqdm import tqdm

# Load data from a CSV file into a PandasDataset
dataset_name = 'USD_CNY Historical Data'
df = pd.read_csv(f'./data/{dataset_name}.csv', parse_dates=['Date'])

df = df.iloc[::-1]
start = df['Date'].iloc[0]
end = df['Date'].iloc[-1]
target = df['Price'].to_numpy()
target = target.reshape(1, -1)

prediction_length = 7
freq = "D"

start = pd.Period(start, freq=freq)  # can be different for each time series
end = pd.Period(end, freq=freq)

# train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
train_ds = ListDataset(
    [{"target": x, "start": start} for x in target[:, :-prediction_length]],
    freq=freq,
)

# test dataset: use the whole dataset, add "target" and "start" fields
test_ds = ListDataset(
    [{"target": x, "start": start} for x in target], freq=freq
)

entry = next(iter(train_ds))
train_series = to_pandas(entry)
# train_series.plot()
# plt.grid(which="both")
# plt.legend(["train series"], loc="upper left")
# plt.show()

entry = next(iter(test_ds))
test_series = to_pandas(entry)
test_series.plot()
plt.axvline(train_series.index[-1], color="r")  # end of train dataset
plt.grid(which="both")
plt.legend(["test series", "end of train series"], loc="upper left")
plt.show()

# Train the model and make predictions
estimator = TemporalFusionTransformerEstimator(
    freq=freq,
    prediction_length=prediction_length,
    context_length=512,
    # quantiles=[0.1, 0.5, 0.9],
    num_heads=32,
    hidden_dim=256,
    variable_dim=128,
    # static_dims=[1, 2, 3],
    # dynamic_dims=[4, 5, 6],
    # past_dynamic_dims=[7, 8, 9],
    # static_cardinalities=[10, 20, 30],
    # dynamic_cardinalities=[40, 50, 60],
    # past_dynamic_cardinalities=[70, 80, 90],
    # lr=0.01,
    trainer_kwargs={"max_epochs": 40}
)
# estimator = DeepAREstimator(
#     prediction_length=prediction_length,
#     num_layers=5,
#     freq=freq,
#     trainer_kwargs={"max_epochs": 20}
# )

predictor = estimator.train(train_ds)
model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if 'type.txt' not in os.listdir(model_path):
    # 创建文件
    with open(os.path.join(model_path, 'type.txt'), 'w') as f:
        pass
predictor.serialize(Path("./model"))

# forecasts = list(model.predict(test_ds))
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)
forecasts = list(tqdm(forecast_it, total=len(test_ds)))

# Plot predictions
display_offset = 30
x = df["Date"][-display_offset:].to_numpy()
x = [pd.Period(p, freq=freq).to_timestamp() for p in x]
plt.plot(x, df["Price"][-display_offset:], color="black")
for forecast in forecasts:
    forecast.start_date = end - prediction_length + 1
    forecast.plot()
plt.legend(["True values"], loc="upper left", fontsize="xx-large")
plt.savefig(f"test_{datetime.datetime.today().strftime('%Y%m%d')}.png")
