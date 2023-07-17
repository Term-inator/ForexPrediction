from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model import Predictor
from tqdm import tqdm

import utils

# Load data from a CSV file into a PandasDataset
dataset_name = 'USD_CNY Historical Data'
df, start, end, target = utils.load_dataset(dataset_name)
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
forcast_start_date = end
utils.display_forcast(df, display_offset, forecasts, forcast_start_date, freq, filename='forcast')
