"""docstring for packages."""
import time
import os
import logging
from datetime import datetime, timedelta
from multiprocessing import Queue
from queue import Empty as EmptyQueueException
import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer
from prometheus_client import Gauge, generate_latest, REGISTRY
from prometheus_api_client import PrometheusConnect, Metric
from configuration import Configuration
from graph_handler import GraphHandler
import schedule

_LOGGER = logging.getLogger(__name__)

PREDICTOR_MODEL_LIST = list()

pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prometheus_headers,
    disable_ssl=True,
)

for metric in Configuration.metrics_list:
    metric_init = pc.get_current_metric_value(metric_name=metric)

    for unique_metric in metric_init:
        PREDICTOR_MODEL_LIST.append(
            Configuration.algorithm(
                unique_metric,
                rolling_data_window_size=Configuration.rolling_training_window_size,
            )
        )

GAUGE_DICT = dict()
for predictor in PREDICTOR_MODEL_LIST:
    unique_metric = predictor.metric
    label_list = list(unique_metric.label_config.keys())
    label_list.append("value_type")
    if unique_metric.metric_name not in GAUGE_DICT:
        GAUGE_DICT[unique_metric.metric_name] = Gauge(
            unique_metric.metric_name + "_" + predictor.model_name,
            predictor.model_description,
            label_list,
        )


class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, data_queue):
        """Check if new predicted values are available in the queue before the get request."""
        try:
            model_list = data_queue.get_nowait()
            self.settings["model_list"] = model_list
        except EmptyQueueException:
            pass

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        for predictor_model in self.settings["model_list"]:
            current_metric_value = Metric(
                pc.get_current_metric_value(
                    metric_name=predictor_model.metric.metric_name,
                    label_config=predictor_model.metric.label_config,
                )[0]
            )

            metric_name = predictor_model.metric.metric_name
            prediction = predictor_model.predict_value(datetime.now() - timedelta(hours=2))

            for column_name in list(prediction.columns):
                GAUGE_DICT[metric_name].labels(
                    **predictor_model.metric.label_config, value_type=column_name
                ).set(prediction[column_name][0])

            anomaly_detector = {
                "more": 0 if current_metric_value.metric_values["y"][0] <
                             prediction["yhat_upper"][0] + Configuration.deviations else 1,
                "less": 0 if current_metric_value.metric_values["y"][0] >
                             prediction["yhat_lower"][0] - Configuration.deviations else 1,
                "both": 0 if prediction["yhat_upper"][0] + Configuration.deviations > current_metric_value.metric_values["y"][0] >
                             prediction["yhat_lower"][0] - Configuration.deviations else 1,
            }

            anomaly = anomaly_detector.get(Configuration.anomaly_border)

            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


def make_app(data_queue):
    """Initialize the tornado web app."""
    _LOGGER.info("Initializing Tornado Web App")
    cwd = os.getcwd()
    return tornado.web.Application(
        [
            (r"/metrics", MainHandler, dict(data_queue=data_queue)),
            (r"/", MainHandler, dict(data_queue=data_queue)),
            (r"/graph", GraphHandler, dict(data_queue=data_queue)),
            (r"/(.*\.html)", tornado.web.StaticFileHandler, {"path": cwd}),
        ]
    )


def train_model(initial_run=False, data_queue=None):
    """Train the machine learning model."""
    for predictor_model in PREDICTOR_MODEL_LIST:
        metric_to_predict = predictor_model.metric
        data_start_time = datetime.now() - Configuration.metric_chunk_size
        if initial_run:
            data_start_time = (
                    datetime.now() - Configuration.rolling_training_window_size
            )

        new_metric_data = pc.get_metric_range_data(
            metric_name=metric_to_predict.metric_name,
            label_config=metric_to_predict.label_config,
            start_time=data_start_time,
            end_time=datetime.now(),
        )[0]

        start_time = datetime.now()
        predictor_model.train(
            new_metric_data, Configuration.retraining_interval_minutes, Configuration.seasonality,
            Configuration.deviations
        )
        _LOGGER.info(
            "Total Training time taken = %s, for metric: %s %s",
            str(datetime.now() - start_time),
            metric_to_predict.metric_name,
            metric_to_predict.label_config,
        )

    data_queue.put(PREDICTOR_MODEL_LIST)


if __name__ == "__main__":
    predicted_model_queue = Queue()

    train_model(initial_run=True, data_queue=predicted_model_queue)

    app = make_app(predicted_model_queue)
    server = HTTPServer(app)
    server.bind(8087)
    server.start()
    tornado.ioloop.IOLoop.current().start()

    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        train_model, initial_run=False, data_queue=predicted_model_queue
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

    server_process.join()
