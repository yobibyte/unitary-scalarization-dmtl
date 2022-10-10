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
RUN pip3.8 install torch==1.7.1+$cuda -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3.8 install torchvision==0.8.2+$cuda -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3.8 install numpy scipy pyyaml matplotlib ruamel.yaml networkx tensorboardX gym tqdm cffi gtimer cffi Cython lockfile
RUN pip3.8 install glfw imageio
RUN pip3.8 install sacred pymongo

ARG uid
ARG user
RUN echo /$user $uid
RUN useradd -d /$user -u $uid $user 
WORKDIR /$user
RUN chmod -R 777 /$user
RUN chmod -R 777 /usr/local

RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
USER $user

RUN mkdir -p /$user/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip --no-check-certificate \
    && unzip mujoco.zip -d /$user/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip --no-check-certificate \
    && unzip mujoco.zip -d /$user/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /$user/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /$user/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /$user/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

RUN pip3.8 install mujoco-py==1.50.1.68

RUN git clone https://github.com/vitchyr/rlkit.git
RUN touch rlkit/rlkit/torch/networks/stochastic/__init__.py
RUN cd rlkit && python3.8 setup.py install

RUN pip3.8 install pybullet
RUN pip3.8 install wandb --user
RUN pip3.8 install sklearn --user

WORKDIR /home/$user/rlmtl
ENV PYTHONPATH /$user/rlmtl

RUN ln -s /$user/.mujoco/mujoco200_linux /$user/.mujoco/mjpro200
RUN pip3.8 install garage --user
ENV PATH /$user/.local/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$user/.mujoco/mujoco200/bin
RUN ln -s /$user/.mujoco/mujoco200_linux /$user/.mujoco/mujoco200
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pip3.8 install pytest --user

WORKDIR /$user
ARG wandb_api_key
ENV WANDB_API_KEY $wandb_api_key

RUN ln -s /usr/bin/python3 /usr/local/bin/python
RUN git clone https://github.com/facebookresearch/mtrl.git && cd mtrl && pip3.8 install -r requirements/dev.txt --user
RUN pip3.8 install git+https://github.com/facebookresearch/mtenv.git@main#egg="mtenv[metaworld]"
RUN pip3.8 install git+https://github.com/rlworkgroup/metaworld.git@af8417bfc82a3e249b4b02156518d775f29eb289
RUN pip3.8 install numpy --user --upgrade
RUN pip3.8 install mtenv
