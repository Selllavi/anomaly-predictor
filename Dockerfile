FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]
ADD . /

ADD environment.yml /tmp/environment.yml
RUN chmod g+w /etc/passwd
RUN conda env create -f /tmp/environment.yml
RUN chmod -R 777 static

CMD /opt/conda/envs/prophet-env/bin/python app.py
