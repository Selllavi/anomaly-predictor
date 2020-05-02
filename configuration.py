"""docstring for installed packages."""
import os
import logging
from prometheus_api_client.utils import parse_datetime, parse_timedelta

if os.getenv("FLT_DEBUG_MODE", "False") == "True":
    LOGGING_LEVEL = logging.DEBUG  # Enable Debug mode
else:
    LOGGING_LEVEL = logging.INFO
# Log record format
logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s: %(message)s", level=LOGGING_LEVEL
)
# set up logging
_LOGGER = logging.getLogger(__name__)


class Configuration:
    """docstring for Configuration."""

    # url for the prometheus host
    prometheus_url = os.getenv("PROMETEUS_URL", "http://prometheus-k8s-monitoring.192.168.99.117.nip.io/")

    # any headers that need to be passed while connecting to the prometheus host
    prometheus_headers = None
    # example oath token passed as a header
    if os.getenv("PROMETEUS_ACCESS_TOKEN"):
        prom_connect_headers = {
            "Authorization": "bearer " + os.getenv("PROMETEUS_ACCESS_TOKEN")
        }

    # list of metrics that need to be scraped and predicted
    # multiple metrics can be separated with a ";"
    # if a metric configuration matches more than one timeseries,
    # it will scrape all the timeseries that match the config.
    metrics_list = str(
        os.getenv(
            "METRICS_LIST",
            "http_requests_total{namespace='default'}",
        )
    ).split(";")

    # this will create a rolling data window on which the model will be trained
    # example: if set to 15d will train the model on past 15 days of data,
    # every time new data is added, it will truncate the data that is out of this range.
    rolling_training_window_size = parse_timedelta(
        "now", os.getenv("ROLLING_TRAINING_WINDOW_SIZE", "2h")
    )

    # How often should the anomaly detector retrain the model (in minutes)
    retraining_interval_minutes = int(
        os.getenv("RETRAINING_INTERVAL_MINUTES", "60")
    )
    metric_chunk_size = parse_timedelta("now", str(retraining_interval_minutes) + "m")

    mlflow_tracking_uri = "http://localhost:5000"

    # threshold value to calculate true anomalies using a linear function
    true_anomaly_threshold = float(os.getenv("TRUE_ANOMALY_THRESHOLD", "0.001"))

    metric_start_time = parse_datetime(os.getenv("DATA_START_TIME", "2020-02-05 13:00:00"))

    metric_end_time = parse_datetime(os.getenv("DATA_END_TIME", "2020-02-05 13:36:00"))

    metric_train_data_end_time = metric_start_time + rolling_training_window_size

    _LOGGER.info("Metric train data start time: %s", metric_start_time)
    _LOGGER.info("Metric train data end time/test data start time: %s", metric_train_data_end_time)
    _LOGGER.info("Metric test end time: %s", metric_end_time)
    _LOGGER.info("Metric data rolling training window size: %s", rolling_training_window_size)
    _LOGGER.info("Model retraining interval: %s minutes", retraining_interval_minutes)
    _LOGGER.info("True anomaly threshold: %s", true_anomaly_threshold)
    _LOGGER.info("MLflow server url: %s", mlflow_tracking_uri)
