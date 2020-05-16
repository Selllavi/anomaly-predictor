"""docstring for packages."""
import time
import os
import io
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, Band
from bokeh.models.tools import HoverTool
from bokeh.resources import CDN
from bokeh.embed import components
from jinja2 import Template
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
import schedule

_LOGGER = logging.getLogger(__name__)

METRICS_LIST = Configuration.metrics_list

PREDICTOR_MODEL_LIST = list()

PREDICTOR_MODEL = Configuration.algorithm

DEVIATIONS = Configuration.deviations

pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prometheus_headers,
    disable_ssl=True,
)

for metric in METRICS_LIST:
    metric_init = pc.get_current_metric_value(metric_name=metric)

    for unique_metric in metric_init:
        PREDICTOR_MODEL_LIST.append(
            PREDICTOR_MODEL(
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

            anomaly = 1
            if (
                    current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0] + DEVIATIONS
            ) and (
                    current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0] - DEVIATIONS
            ):
                anomaly = 0

            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


class GraphHandler(tornado.web.RequestHandler):
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
            df = predictor_model.predicted_df

            output_file('static/filename.html')
            TOOLS = "pan,wheel_zoom,reset,save"
            plot = figure(plot_width=800, plot_height=300, x_axis_type="datetime", title=predictor_model.model_name,
                          tools=TOOLS, toolbar_location="above")
            plot.xaxis.axis_line_color = "rgb(173, 173, 173)"
            plot.yaxis.axis_line_color = "white"
            plot.yaxis.major_tick_line_color = "white"
            plot.xaxis.major_tick_in = 1

            metric_values = predictor_model.metric.metric_values
            source_real = ColumnDataSource(data=dict(values=metric_values.values[:, 1], dates=metric_values.values[:, 0]))
            source_yhat = ColumnDataSource(data=dict(values=df.yhat.values, dates=df.yhat.index.values))
            source_yhat_bound = ColumnDataSource(data=dict(yhat_upper=df.yhat_upper.values, yhat_lower=df.yhat_lower.values,dates=df.yhat_upper.index.values))
            band = Band(base='dates', lower='yhat_lower', upper='yhat_upper', source=source_yhat_bound,
                        level='underlay', fill_alpha=0.5, fill_color="rgb(200, 200, 200)", line_width=0)
            plot.add_layout(band)

            plot.line(x='dates', y='values', source=source_yhat, color="red")
            plot.line(x='dates', y='values', source=source_real, color="aquamarine")
            plot.xgrid.visible = False
            plot.title.text_color = "rgb(173, 173, 173)"
            plot.yaxis.minor_tick_line_color = None
            plot.add_tools(HoverTool(
                tooltips=[
                    ('date', '@dates{|%F %T|}'),
                    ('value', '$y'),
                ],
                formatters={
                    '@dates': 'datetime'
                },
                mode='vline'
            ))

            script_bokeh, div_bokeh = components(plot)
            resources_bokeh = CDN.render()

            template = Template(
                '''<!DOCTYPE html>
                    <html lang="en">
                        <head>
                            <meta charset="utf-8">
                            <title>Overview</title>
                            {{ resources }}
                            {{ script }}
                            <style>
                                .embed-wrapper {
                                    display: flex;
                                    justify-content: space-evenly;
                                }
                                .bk-logo {
                                    display:none !important;
                                }
                                .bk-tool-icon-hover {
                                    display:none !important;
                                }
                            </style>
                        </head>
                        <body style="background-color:rgb(246, 246, 246);>                  
                            <div class="embed-wrapper">
                                {{ div }}
                            </div>
                        </body>
                    </html>
                    ''')

            html = template.render(resources=resources_bokeh,
                                   script=script_bokeh,
                                   div=div_bokeh)

            out_file_path = "static/filename.html"
            with io.open(out_file_path, mode='w') as f:
                f.write(html)
        self.render("static/filename.html")


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
            new_metric_data, Configuration.retraining_interval_minutes, Configuration.seasonality
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
