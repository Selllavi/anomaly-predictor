"""doctsring for packages."""
import datetime
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import fft
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prometheus_api_client import Metric

# Set up logging
_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "Sarima"
    model_description = "Prediction values based on Sarima procedure"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

    def train(self, metric_data=None, prediction_duration=15, seasonality=None):
        """Train the Sarima model and store the predictions in predicted_df."""
        prediction_freq = "1MIN"
        # convert incoming metric to Metric Object

        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained

        data = self.metric.metric_values
        values = pd.Series(self.metric.metric_values.y.values, index = data["ds"])
        self.model = SARIMAX(values, order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12), enforce_stationarity = True, enforce_invertibility = False)

        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")
        results = self.model.fit(dsip=-1)
        forecast = results.forecast(prediction_duration)
        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast)

        # find most recent timestamp from original data and extrapolate new timestamps
        _LOGGER.debug("Creating Dummy Timestamps.....")
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast), freq="30s"
        )

        # create dummy upper and lower bounds
        _LOGGER.debug("Computing Bounds .... ")

        upper_bound = np.array(
            [
                (
                        np.ma.average(
                            forecast[:i],
                            weights=np.linspace(0, 1, num=len(forecast[:i])),
                        )
                        + (np.std(forecast[:i]) * 2)
                )
                for i in range(len(forecast))
            ]
        )
        upper_bound[0] = np.mean(
            forecast
        )  # to account for no std of a single value
        lower_bound = np.array(
            [
                (
                        np.ma.average(
                            forecast[:i],
                            weights=np.linspace(0, 1, num=len(forecast[:i])),
                        )
                        - (np.std(forecast[:i]) * 2)
                )
                for i in range(len(forecast))
            ]
        )
        lower_bound[0] = np.mean(
            forecast
        )  # to account for no std of a single value
        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound

        # create series and index into predictions_dict
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
