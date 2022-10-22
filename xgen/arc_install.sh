#/bin/bash
container="ubuntu"
# lxd init

lxc launch ubuntu-daily:18.04 ubuntu
lxc start ubuntu

lxc config device add ubuntu gpu gpu # Add GPU device
lxc exec ubuntu -- sudo apt-get install gcc linux-headers-generic unzip curl git docker.io -y
lxc exec ubuntu -- sudo apt-get upgrade -y
lxc exec ubuntu -- uname -r
lxc exec ubuntu -- wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
lxc exec ubuntu -- sudo dpkg -i cuda-keyring_1.0-1_all.deb
lxc exec ubuntu -- sudo apt-get update
lxc exec ubuntu -- sudo apt-get install cuda -y
lxc exec ubuntu -- apt dist-upgrade -y
lxc exec ubuntu -- wget http://xgen-install.cocopie.ai/xgen_docker_deploy.sh
lxc exec ubuntu -- curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash   # to install git-lfs
lxc exec ubuntu -- sudo apt-get install git-lfs -y
lxc exec ubuntu -- bash xgen_docker_deploy.sh --install # Run XGen Installation.
