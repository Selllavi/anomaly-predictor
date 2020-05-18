"""doctsring for packages."""
import logging
import pandas as pd
import numpy as np
import common_tools as ct
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prometheus_api_client import Metric

_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "agile"
    model_description = "Prediction values based on Sarima procedure"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

    def train(self, metric_data=None, prediction_duration=15, seasonality=None, deviations=3):
        """Train the Sarima model and store the predictions in predicted_df."""
        if metric_data:
            self.metric += Metric(metric_data)

        data = self.metric.metric_values
        values = pd.Series(self.metric.metric_values.y.values, index = data["ds"])
        self.model = SARIMAX(values, order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12), enforce_stationarity = True, enforce_invertibility = False)

        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        _LOGGER.debug("begin training")
        results = self.model.fit(dsip=-1)
        forecast = results.forecast(prediction_duration)
        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast)

        _LOGGER.debug("Creating Dummy Timestamps.....")
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast), freq="30s"
        )

        _LOGGER.debug("Computing Bounds .... ")

        lower_bound, upper_bound = ct.calculate_bounds(forecast, deviations)

        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound
        _LOGGER.debug("Formatting Forecast to Pandas ..... ")

        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index("timestamp")

        self.predicted_df = forecast
        _LOGGER.debug(forecast)

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        nearest_index = self.predicted_df.index.get_loc(
            prediction_datetime, method="nearest"
        )
        return self.predicted_df.iloc[[nearest_index]]
