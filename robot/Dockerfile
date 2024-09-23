# docker build . -t act_trossen_arm -f robot/Dockerfile
# docker run --rm -it --network=host -v /dev:/dev --privileged -e ARM_ROLE=master -e ARM_SIDE=left -e ARM_SERIAL_ID=FT8J0VCM act_trossen_arm /bin/bash

FROM ros:noetic-robot@sha256:0e12e4db836e78c74c4b04c6d16f185d9a18d2b13cf5580747efa075eb6dc6e0
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    whiptail \
    git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends keyboard-configuration

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
RUN chmod +x xsarm_amd64_install.sh
RUN export DEBIAN_FRONTEND=noninteractive && export TZ='America/Los_Angeles' && ./xsarm_amd64_install.sh -d noetic -n

RUN cd /root/interbotix_ws/src && git clone https://github.com/tonyzhaozh/aloha.git

RUN apt-get update && \
    apt-get install -y \
    cmake \
    openssh-client \
    ros-noetic-cv-bridge \
    ros-noetic-usb-cam \
    && pip install \
    pyquaternion \
    pyyaml \
    rospkg \
    pexpect \
    mujoco==2.3.7 \
    dm_control==1.0.14 \
    opencv-python \
    matplotlib \
    einops \
    packaging \
    h5py \
    IPython

COPY ./robot/master_config.yaml /root/interbotix_ws/src/monopi/master_config.yaml
COPY ./robot/puppet_config.yaml /root/interbotix_ws/src/monopi/puppet_config.yaml
# COPY ./robot/arm.py /root/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py
RUN cd /root/interbotix_ws && source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && catkin_make

CMD ["/bin/bash", "-c", \
    "source /opt/ros/noetic/setup.sh && \
    source /root/interbotix_ws/devel/setup.sh && \
    sed -i \"s/\\$ARM_SERIAL_ID/$ARM_SERIAL_ID/g\" /root/interbotix_ws/src/monopi/${ARM_ROLE}_config.yaml && \
    roslaunch --wait \
    /root/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_control/launch/xsarm_control.launch \
    robot_model:=wx250s \
    robot_name:=${ARM_ROLE}_${ARM_SIDE} \
    base_link_frame:=base_link \
    use_world_frame:=false \
    use_rvis:=false \
    mode_configs:=/root/interbotix_ws/src/monopi/${ARM_ROLE}_config.yaml \
    use_sim:=false"]