FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libnvidia-gl-535 \
    cmake \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

# COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install --upgrade pip
RUN pip install ninja imageio imageio-ffmpeg

COPY nvdiffrast/nvdiffrast /tmp/pip/nvdiffrast/
COPY nvdiffrast/README.md nvdiffrast/setup.py /tmp/pip/
RUN cd /tmp/pip && pip install .

# Install additional dependencies from pypi
COPY requirements.txt /usr/share/requirements.txt
RUN pip install -r /usr/share/requirements.txt

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
