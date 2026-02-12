#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

FROM ubuntu:22.04@sha256:ed1544e454989078f5dec1bfdabd8c5cc9c48e0705d07b678ab6ae3fb61952d2
ARG SERVER_GPU

RUN echo $HTTP_PROXY
RUN if [ -n "$HTTP_PROXY" ] ; then  echo "Acquire::http::Proxy \"$HTTP_PROXY\";" >  /etc/apt/apt.conf; fi
RUN apt-get update -y || true; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    git \
    iotop \
    sysstat \
    jq \
    curl \
    cmake \
    python3-pip \
    build-essential \
    docker.io \
    pciutils

COPY requirements.txt .

RUN pip install --require-hashes --no-deps --no-cache-dir -r requirements.txt

RUN if [ -d "/opt/intel/pcm" ] ; then rm -R /opt/intel/pcm; fi

ENV PCM_DIRECTORY=/opt/intel
RUN echo "Installing PCM" \
    [ ! -d "$PCM_DIRECTORY" ] && mkdir -p "$PCM_DIRECTORY"
RUN cd $PCM_DIRECTORY && \
    git clone --recursive https://github.com/opcm/pcm.git && \
    ls ${PCM_DIRECTORY} && \
    cd $PCM_DIRECTORY/pcm  && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build .

# Cleanup
RUN mkdir -p "/opt/intel/pcm-bin/bin" && mkdir -p "/opt/intel/pcm-bin/lib" && \
    cp -r "$PCM_DIRECTORY/pcm/build/bin" "/opt/intel/pcm-bin/" && \
    cp -r "$PCM_DIRECTORY/pcm/build/lib" "/opt/intel/pcm-bin/" && \
    rm -rf "$PCM_DIRECTORY/pcm"

COPY entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]