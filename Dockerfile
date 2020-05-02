FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]
ADD . /

ADD environment.yml /tmp/environment.yml
RUN chmod g+w /etc/passwd
RUN conda env create -f /tmp/environment.yml

python app.py
