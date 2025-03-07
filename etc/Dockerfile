FROM continuumio/miniconda3

# This should be built from the root WISER directory

RUN apt-get update && apt-get install -y make

COPY /etc /WISER-etc

RUN conda env create -f WISER-etc/wiser-source.yml

RUN conda init

SHELL ["conda", "run", "-n", "wiser-source", "/bin/bash", "-c"]

ENV PATH /opt/conda/envs/myenv/bin:$PATH

RUN pip install --no-cache-dir -r WISER-etc/requirements.txt

RUN Xvfb :1 -screen 0 1024x768x16 &

RUN export DISPLAY=:1
