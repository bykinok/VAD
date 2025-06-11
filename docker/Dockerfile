# 대안들:
# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 1. 안정적인 CUDA 11.3 + Ubuntu 20.04 베이스 이미지 사용 (확실히 존재하는 버전)
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 2. DNS 설정 및 네트워크 문제 해결
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# 3. 환경 변수 설정 (비대화형 설치)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 4. 패키지 소스 목록 업데이트 및 재시도 로직
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing || \
    (sleep 10 && apt-get update --fix-missing) || \
    (sleep 30 && apt-get update --fix-missing)

# 5. 시스템 필수 패키지 및 Python 설치 (재시도 로직 포함)
RUN apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    wget \
    curl \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    ca-certificates \
    software-properties-common \
    net-tools \
    iputils-ping \
    ninja-build \
    gcc \
    g++ \
    || (apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    wget \
    curl \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    ca-certificates \
    software-properties-common \
    net-tools \
    iputils-ping \
    ninja-build \
    gcc \
    g++) \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 5.5. GUI Visualization을 위한 X11 및 OpenGL 패키지 설치
RUN apt-get update && apt-get install -y \
    x11-apps \
    python3-tk \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    libice6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 6. Python 3.8을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 6.5. CUDA 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}
ENV CPATH=${CUDA_HOME}/include:${CPATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 6.6. GUI Visualization을 위한 환경 변수 설정
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330
ENV LIBGL_ALWAYS_SOFTWARE=1

# 6.7. X11 및 GUI 지원을 위한 디렉토리 생성
RUN mkdir -p /tmp/runtime-root && \
    chmod 755 /tmp/runtime-root

# 7. pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# 7.5. cuDNN 설치 (CUDA 11.1 호환)
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 8. PyTorch 및 torchvision 설치 (CUDA 11.1 호환 버전, THC 헤더 호환성을 위해 1.9.1 사용)
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# 9. mmcv-full 설치 (PyTorch 1.9.1 + CUDA 11.1 호환 버전)
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 10. mmdet 및 mmseg 설치
RUN pip install mmdet==2.14.0 && \
    pip install mmsegmentation==0.14.1

# 11. timm 설치
RUN pip install timm

# 12. nuscenes-devkit 설치
RUN pip install nuscenes-devkit==1.1.9

# 13. 작업 디렉토리 설정
WORKDIR /workspace

# 14. mmdetection3d 설치 전 필수 패키지 설치
RUN pip install scikit-image==0.19.3 trimesh==3.15.8

# 14.1. mmdetection3d 설치
RUN git clone https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && \
    git checkout -f v0.17.1 && \
    pip install -v -e .

# 15. VAD 리포지토리 클론
RUN git clone https://github.com/bykinok/VAD.git

# 16. 사전 훈련된 모델 다운로드
RUN cd VAD && \
    mkdir ckpts && \
    cd ckpts && \
    wget https://download.pytorch.org/models/resnet50-19c8e357.pth

# 17. 작업 디렉토리를 VAD로 변경
WORKDIR /workspace/VAD

# 18. 컨테이너 시작 시 기본 명령어
CMD ["python"]

# ------------------------------------------------------------------------------------------------
# 컨테이너 build 명령어
# docker build --network=host -t vad .

# 컨테이너 실행 명령어
# docker run --gpus all --dns=8.8.8.8 --dns=8.8.4.4 --shm-size=16g -it -v /home/mando/DATA:/NAS vad /bin/bash

# VAD 학습 명령어 예시
# python tools/train.py configs/VAD/VAD_tiny_stage1.py --work-dir work_dirs/VAD_tiny_stage1

# VAD 평가 명령어 예시  
# python tools/test.py configs/VAD/VAD_tiny_stage2.py work_dirs/VAD_tiny_stage2/latest.pth --eval bbox류