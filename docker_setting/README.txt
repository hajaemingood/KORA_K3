0) 사전 체크(호스트)
# NVIDIA 툴킷
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# X11 권한 (GUI용)
xhost +local:docker

1) build
docker compose up -d --build

2) 접속
docker exec -it kora_k3 bash

3) 컨테이너 내부 Follow-The-Gap 시뮬레이터 실행 순서

  # ROS 환경 로드
  source /opt/ros/noetic/setup.bash

  # 의존 패키지 설치 (최초 1회)
  apt update
  apt install -y \
    ros-noetic-ackermann-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-rviz

  # 워크스페이스
  cd /root/KORA_K3
  catkin_make
  source devel/setup.bash

  # 시뮬레이터 패키지가 없다면 클론 (최초 1회)
  if [ ! -d src/f1tenth_simulator ]; then
    cd src
    git clone https://github.com/f1tenth/f1tenth_simulator.git
    cd ..
    catkin_make
    source devel/setup.bash
  fi

  # 터미널 1: 시뮬레이터 구동
  roslaunch f1tenth_simulator simulator.launch

  # 터미널 2: FTG 노드 실행
  source /root/KORA_K3/devel/setup.bash
  rosrun kora_k3 ftg_node_sim.py

  # 상태 확인
  rostopic echo /drive   # 속도/조향 명령
  rostopic echo /scan    # LiDAR 데이터
