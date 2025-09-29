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
