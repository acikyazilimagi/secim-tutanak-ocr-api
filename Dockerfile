FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ARG PYTHON_VERSION=3.8
RUN apt-get install -y wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION}
RUN apt-get install -y python3-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN pip3 install --upgrade pip

ENV FORCE_CUDA="1"

COPY ./requirements.txt /code/requirements.txt

WORKDIR /code

#install paddlepaddle for cuda 11.2
RUN python3 -m pip install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
RUN pip3 install -r /code/requirements.txt

COPY . .

EXPOSE 8080
#CMD ["uvicorn", "secim_tutanak_ocr_api.main:app", "--host", "0.0.0.0", "--port", "8080"]