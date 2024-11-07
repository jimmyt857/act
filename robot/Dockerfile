# docker build . -t act_trossen_arm -f robot/Dockerfile
# docker run --rm -it --network=host -v /dev:/dev --privileged act_trossen_arm /bin/bash

FROM ros:noetic-robot@sha256:0e12e4db836e78c74c4b04c6d16f185d9a18d2b13cf5580747efa075eb6dc6e0
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    curl \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    whiptail \
    git \
    wget \
    openssh-client \
    ros-noetic-cv-bridge \
    ros-noetic-usb-cam \
    keyboard-configuration

WORKDIR /root
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
RUN chmod +x xsarm_amd64_install.sh
RUN export DEBIAN_FRONTEND=noninteractive && export TZ='America/Los_Angeles' && ./xsarm_amd64_install.sh -d noetic -n

# TODO: is this needed?
# RUN cd /root/interbotix_ws/src && git clone https://github.com/tonyzhaozh/aloha.git

COPY ./robot/master_config.yaml /root/interbotix_ws/src/openpi/master_config.yaml
COPY ./robot/puppet_config.yaml /root/interbotix_ws/src/openpi/puppet_config.yaml
COPY ./robot/arm.py /root/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py
RUN cd /root/interbotix_ws && source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && catkin_make

RUN git clone https://github.com/tonyzhaozh/aloha.git && \
    cd aloha && \
    git checkout 06369f03cd8e0a47e16d3a90167853fd33af7557 && \
    sed -i 's/<put your data dir here>/\/data/g' ./aloha_scripts/constants.py

RUN git clone https://github.com/jimmyt857/act.git && \
    cd act && \
    git checkout f07d0ec38bab32df8169638c1fab86a8ea7915c6

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda env create -n aloha -f /root/act/conda_env.yaml python=3.8.10
RUN source /opt/conda/bin/activate aloha && conda install -y mkl==2024.0 && pip install modern_robotics
ENV PYTHONPATH=$PYTHONPATH:/root/aloha/aloha_scripts:/root/aloha:/root/act/detr

CMD ["/bin/bash", "-c", \
    "source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && source /opt/conda/bin/activate aloha && \
    python /root/act/imitate_episodes.py --task_name aloha_wear_shoe --ckpt_dir /ckpts --eval --policy_class PI --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --render_height 256 --render_width 320"]
