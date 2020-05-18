"""docstring for installed packages."""
import logging
import pandas as pd
import numpy as np
from prometheus_api_client import Metric
import common_tools as ct
from numpy import fft

_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "basic"
    model_description = "Forecast value based on fourier analysis"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

    def fourier_extrapolation(self, input_series, n_predict, n_harmonics):
        """Perform the Fourier extrapolation on time series data."""
        n = input_series.size
        t = np.arange(0, n)
        p = np.polyfit(t, input_series, 1)
        input_no_trend = input_series - p[0] * t
        frequency_domain = fft.fft(input_no_trend)
        frequencies = fft.fftfreq(n)
        indexes = np.arange(n).tolist()
        indexes.sort(key=lambda i: np.absolute(frequencies[i]))

        time_steps = np.arange(0, n + n_predict)
        restored_signal = np.zeros(time_steps.size)

        for i in indexes[: 1 + n_harmonics * 2]:
            amplitude = np.absolute(frequency_domain[i]) / n
            phase = np.angle(frequency_domain[i])
            restored_signal += amplitude * np.cos(
                2 * np.pi * frequencies[i] * time_steps + phase
            )

        restored_signal = restored_signal + p[0] * time_steps
        return restored_signal[n:]

    def train(self, metric_data=None, prediction_duration=15, seasonality=None, deviations=3):
        """Train the Fourier model and store the predictions in pandas dataframe."""
        prediction_range = prediction_duration
        if metric_data:
            self.metric += Metric(metric_data)

        data = self.metric.metric_values
        vals = np.array(data["y"].tolist())

        _LOGGER.debug("training data start time: %s", self.metric.start_time)
        _LOGGER.debug("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")

        forecast_values = self.fourier_extrapolation(
            vals, prediction_range, 1
        )
        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast_values)

        _LOGGER.debug("Creating Dummy Timestamps.....")
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="30s"
        )

        _LOGGER.debug("Calculating Bounds .... ")

        lower_bound, upper_bound = ct.calculate_bounds(forecast_values, deviations)

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
