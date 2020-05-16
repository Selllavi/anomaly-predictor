FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]
ADD . /

ADD environment.yml /tmp/environment.yml
RUN chmod -R 777 static
RUN chmod g+w /etc/passwd
RUN conda env create -f /tmp/environment.yml

CMD /opt/conda/envs/prophet-env/bin/python app.py
