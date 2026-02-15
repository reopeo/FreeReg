FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip \
    git wget curl ca-certificates \
    build-essential cmake ninja-build \
    libopenblas-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python \
 && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip \
 && pip install uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv ${VIRTUAL_ENV} --python python3.8
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements-pytorch.txt /workspace/requirements-pytorch.txt
COPY requirements.txt /workspace/requirements.txt

RUN uv pip install -r /workspace/requirements-pytorch.txt

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN uv pip install -r /workspace/requirements.txt

ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN git clone --recursive https://github.com/NVIDIA/MinkowskiEngine.git /opt/MinkowskiEngine \
 && cd /opt/MinkowskiEngine \
 && export CUDA_HOME=$(dirname $(dirname $(which nvcc))) \
 && python setup.py install --force_cuda --blas=openblas --blas_include_dirs=/usr/include

RUN uv pip uninstall opencv-python opencv-python-headless \
 && uv pip install opencv-python-headless==4.5.5.64

WORKDIR /workspace
CMD ["/bin/bash"]
