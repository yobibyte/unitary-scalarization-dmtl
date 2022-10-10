FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG cuda

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
ENV CPATH /usr/local/cuda/include
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 7.5+PTX"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils vim man build-essential wget sudo python3.8 python3-pip htop zlib1g-dev swig unzip libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf git-all python3.8-dev&& \
    rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip
RUN pip3.8 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3.8 install numpy scipy pyyaml matplotlib ruamel.yaml networkx tensorboardX gym tqdm cffi gtimer cffi Cython lockfile
RUN pip3.8 install glfw imageio
RUN pip3.8 install sacred pymongo
RUN pip3.8 install wandb 
RUN pip3.8 install sklearn

ARG uid
ARG user
RUN echo /$user $uid
RUN useradd -d /$user -u $uid $user 
USER $user


WORKDIR /$user/mtrlsupervised
ENV PYTHONPATH /$user/mtrlsupervised
ARG wandb_api_key
ENV WANDB_API_KEY $wandb_api_key

ENV PATH /$user/.local/bin:$PATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV HOME /$user/mtrlsupervised

