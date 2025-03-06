FROM continuumio/miniconda3

WORKDIR /WISER

COPY . /WISER

RUN conda env create -f etc/wiser-source.yml

RUN conda init

SHELL ["/bin/bash", "-c"]

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate wiser-source

ENV PATH /opt/conda/envs/myenv/bin:$PATH

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate wiser-source && pip install --no-cache-dir -r etc/requirements-test3.txt

RUN Xvfb :1 -screen 0 1024x768x16 &

RUN export DISPLAY=:1

RUN rm -rf /WISER
