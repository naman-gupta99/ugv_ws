ARG IMAGE="osrf/ros:humble-desktop-full"

FROM ${IMAGE}

ARG WS_ROS
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV USER=root
ENV WS=/${WS_ROS}
WORKDIR ${WS}

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    unzip \
    git \
    nano \
    graphviz \
    iputils-ping \
    net-tools

# Install ROS2 Humble-specific packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-* \
    ros-${ROS_DISTRO}-rosidl-generator-py \
    ros-${ROS_DISTRO}-rosbridge-suite \
    ros-${ROS_DISTRO}-libg2o \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-gazebo-* \
    ros-${ROS_DISTRO}-rtabmap-ros \
    libconsole-bridge-dev \
    libspdlog-dev \
    libfmt-dev

# Clone Gazebo models into the linked workspace
RUN git clone https://github.com/osrf/gazebo_models.git ~/.gazebo/models

# Add alias for Python argcomplete (no need to source during build)
RUN echo 'alias register-python-argcomplete="register-python-argcomplete3"' >> ~/.bashrc

# Upgrade all packages
RUN apt-get upgrade -y

# Set up environment
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "export GAZEBO_MODEL_PATH=~/.gazebo/models" >> ~/.bashrc

# Default command
CMD ["bash"]