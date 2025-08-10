FROM nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    # Include Pascal, Volta, Turing, and Ampere architectures for SVOX build
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
    # Colmap variables
    CUDA_ARCHITECTURES=native \
    COLMAP_GIT_COMMIT=main \
    PYTHONPATH=/queen/submodules/simple-knn

RUN apt-get update && \
    apt-get install --no-install-recommends -y wget \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        zip \
        ninja-build \
        ffmpeg \
        python3.11 \
        python3.11-dev \
        # Colmap dependencies
        cmake \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libmetis-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        # 4DGaussians dependencies
        python3.11-tk \
        # queen dependencies
        libgl1 \
        libglm-dev && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python and python3 to point to python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install uv. TODO: Port from pip to uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh
# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -m pip install --upgrade pip

# https://github.com/colmap/colmap/blob/1f7f28ae1b282286cdd243d14e600f39fdde60d4/docker/Dockerfile#L41
RUN git clone https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout 3.11.1 && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja install

# https://github.com/JimothyJohn/4DGaussians?tab=readme-ov-file#environmental-setups
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade setuptools wheel && \
    git clone --recursive https://github.com/NVlabs/queen /queen && \
    cd /queen && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip install --no-build-isolation --use-pep517 -e submodules/diff-gaussian-rasterization && \
    python -m pip install --no-build-isolation --use-pep517 -e submodules/gaussian-rasterization-grad && \
    # Fix error: "submodules/simple-knn/simple_knn.cu(154): error: identifier "FLT_MAX" is undefined"
    sed -i '/#include <cooperative_groups\/reduce.h>/a #include <cfloat>' /queen/submodules/simple-knn/simple_knn.cu && \
    python -m pip install --no-build-isolation --use-pep517 -e /queen/submodules/simple-knn && \
    mv /queen/maxxvit.py /usr/local/lib/python3.11/dist-packages/timm/models/maxxvit.py && \
    sed -i "s/os.path.join(datadir,'cam00','images','0000.png')/os.path.join(datadir,'cam01','images','0000.png')/g" /queen/scene/dataset_readers.py
