# syntax=docker/dockerfile:1

# This is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN --mount=type=ssh git clone git@github.com:pcr-upm/opal23_headpose.git opal23_headpose
ADD data /opal23_headpose/data

# Copy the repository from the previous image
FROM nvcr.io/nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ENV LANG=C.UTF-8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget libgl1-mesa-glx libsm6 libxext6 libglib2.0-0
RUN mkdir -p /home/username
WORKDIR /home/username
COPY --from=intermediate /opal23_headpose /home/username/opal23_headpose
LABEL maintainer="roberto.valle@upm.es"
# Setup conda environment
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/username/miniconda.sh
RUN chmod +x /home/username/miniconda.sh
RUN /home/username/miniconda.sh -b -p /home/username/conda
RUN /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /home/username/conda/bin/conda create --name opal23 python=3.10
# Activate conda environment
ENV PATH /home/username/conda/envs/opal23/bin:/home/username/conda/bin:$PATH
# Make RUN commands use the new environment (source activate opal23)
SHELL ["conda", "run", "-n", "opal23", "/bin/bash", "-c"]
# Install dependencies
RUN pip install pcr-framework tqdm "numpy==1.23.5"
RUN pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
