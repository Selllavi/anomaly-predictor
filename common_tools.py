"""docstring for installed packages."""
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


def calculate_bounds(forecast_values, deviations):
    average = calculate_average(forecast_values)
    lower_bound = np.array(
        [
            (
                    average[i]
                    - (np.std(forecast_values[:i]) * deviations)
            )
            for i in range(len(forecast_values))
        ]
    )
    upper_bound = np.array(
        [
            (
                    average[i]
                    + (np.std(forecast_values[:i]) * deviations)
            )
            for i in range(len(forecast_values))
        ]
    )
    upper_bound[0] = np.mean(
        forecast_values
    )
    lower_bound[0] = np.mean(
        forecast_values
    )
    return lower_bound, upper_bound


def calculate_average(forecast_values):
    average = np.array(
        [
            (
                np.ma.average(
                    forecast_values[:i],
                    weights=np.linspace(0, 1, num=len(forecast_values[:i])),
                )
            )
            for i in range(len(forecast_values))
        ]
    )
    return average
