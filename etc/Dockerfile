FROM continuumio/miniconda3

# This should be built from the root WISER directory

RUN apt-get update && apt-get install -y make

COPY /etc /WISER-etc

RUN conda env create -f WISER-etc/wiser-source.yml

RUN conda init

SHELL ["conda", "run", "-n", "wiser-source", "/bin/bash", "-c"]

ENV PATH /opt/conda/envs/myenv/bin:$PATH

RUN apt-get update && apt-get install -y xvfb

ENTRYPOINT [ "WISER-etc/entry_point.sh" ]
