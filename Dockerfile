FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Taken from ROS 2 Dockerfiles

# ros-core

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN set -eux; \
       key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
       export GNUPGHOME="$(mktemp -d)"; \
       gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
       mkdir -p /usr/share/keyrings; \
       gpg --batch --export "$key" > /usr/share/keyrings/ros2-latest-archive-keyring.gpg; \
       gpgconf --kill all; \
       rm -rf "$GNUPGHOME"

# setup sources.list
RUN echo "deb [ signed-by=/usr/share/keyrings/ros2-latest-archive-keyring.gpg ] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO humble

# install ros2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-ros-core=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# ros-base

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# ros-desktop

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop=0.10.0-1* \
    ros-humble-sensor-msgs-py \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-tf-transformations \
    ros-humble-ament-cmake-nose \
    && rm -rf /var/lib/apt/lists/*

# copy source code:
COPY . /tmp/rosbag2_language_sam

# installing python dependencies:
WORKDIR $HOME/rosbag2_language_sam

RUN apt-get update && apt-get install -y nano python-is-python3

# We need cuda home before installing the deps
ENV CUDA_HOME="/usr/local/cuda"
RUN printenv | grep CUDA
RUN CUDA_HOME=/usr/local/cuda pip install -r /tmp/rosbag2_language_sam/requirements.txt
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /GroundingDINO
RUN cd /GroundingDINO && pip install -e .

RUN ln -s /rosbag2_language_sam/.cache/huggingface /root/.cache/huggingface 
RUN ln -s /rosbag2_language_sam/.cache/torch /root/.cache/torch 

ENTRYPOINT ["/tmp/rosbag2_language_sam/entrypoint.sh"]