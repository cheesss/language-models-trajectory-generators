# Ubuntu 20.04 이미지를 베이스로 사용
FROM ubuntu:20.04

# 설치 중 인터랙티브 프롬프트 방지
ENV DEBIAN_FRONTEND=noninteractive

# 기본 시스템 업데이트 및 필요한 도구 설치
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      python3-pip \
      build-essential \
      libgl1-mesa-glx \
      libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*


# python3와 python 명령어를 Python 3.10으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# get-pip.py를 이용하여 최신 pip 설치 (시스템 pip 대신 최신 pip 사용)
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사
COPY requirements.txt /app

# requirements.txt에 따른 패키지 설치 (pyrealsense2 포함)
RUN python -m pip install -r requirements.txt

# 전체 프로젝트 소스 복사
COPY . /app

# 기본 CMD를 bash로 설정 (컨테이너 실행 후 수동으로 'python main.py --robot franka' 실행 가능)
CMD ["bash"]
