import io
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, Band
from bokeh.models.tools import HoverTool
from bokeh.resources import CDN
from bokeh.embed import components
from jinja2 import Template
from queue import Empty as EmptyQueueException
import tornado.web


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

            output_file('static/graph.html')
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
                        level='overlay', fill_alpha=0.7, fill_color="rgb(200, 200, 200)", line_width=0)
            plot.add_layout(band)

            plot.line(x='dates', y='values', source=source_yhat, color="blue")
            plot.line(x='dates', y='values', source=source_real, color="red")
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

            out_file_path = "static/graph.html"
            with io.open(out_file_path, mode='w') as f:
                f.write(html)
        self.render("static/graph.html")