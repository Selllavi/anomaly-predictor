"""docstring for installed packages."""
import os
import logging
import fourier_model
import prophet_model
import sarima_model
from prometheus_api_client.utils import parse_datetime, parse_timedelta

if os.getenv("FLT_DEBUG_MODE", "False") == "True":
    LOGGING_LEVEL = logging.DEBUG
else:
    LOGGING_LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s: %(message)s", level=LOGGING_LEVEL
)

_LOGGER = logging.getLogger(__name__)


class Configuration:
    """docstring for Configuration."""

    prometheus_url = os.getenv("PROMETEUS_URL", "http://prometheus-k8s-monitoring.192.168.99.101.nip.io")

    prometheus_headers = None
    if os.getenv("PROMETEUS_ACCESS_TOKEN"):
        prom_connect_headers = {
            "Authorization": "bearer " + os.getenv("PROMETEUS_ACCESS_TOKEN")
        }

    metrics_list = str(
        os.getenv(
            "METRICS_LIST",
            "http_request_duration_microseconds{endpoint='https',handler='prometheus',instance='10.0.2.15:8443',job='apiserver',namespace='default',quantile='0.99',service='kubernetes'}",
        )
    ).split(";")

    rolling_training_window_size = parse_timedelta(
        "now", os.getenv("ROLLING_TRAINING_WINDOW_SIZE", "2m")
    )

    retraining_interval_minutes = int(
        os.getenv("RETRAINING_INTERVAL_MINUTES", "60")
    )
    metric_chunk_size = parse_timedelta("now", str(retraining_interval_minutes) + "m")

    deviations = int(
        os.getenv("DEVIATIONS", "3")
    )

    algorithm_name = str(
        os.getenv("ALGORITHM", "basic")
    )

    algorithm_resolver = {
        "robust": prophet_model.MetricPredictor,
        "agile": sarima_model.MetricPredictor,
        "basic": fourier_model.MetricPredictor
    }
    algorithm = algorithm_resolver.get(algorithm_name, "agile")

    seasonality = str(
        os.getenv("SEASONALITY", "daily")
    )

    mlflow_tracking_uri = "http://localhost:5000"

    metric_start_time = parse_datetime(os.getenv("DATA_START_TIME", "2020-02-05 13:00:00"))

    metric_end_time = parse_datetime(os.getenv("DATA_END_TIME", "2020-02-05 13:36:00"))

    metric_train_data_end_time = metric_start_time + rolling_training_window_size

    _LOGGER.info("Metric train data start time: %s", metric_start_time)
    _LOGGER.info("Metric train data end time/test data start time: %s", metric_train_data_end_time)
    _LOGGER.info("Metric test end time: %s", metric_end_time)
    _LOGGER.info("Metric data rolling training window size: %s", rolling_training_window_size)
    _LOGGER.info("Model retraining interval: %s minutes", retraining_interval_minutes)