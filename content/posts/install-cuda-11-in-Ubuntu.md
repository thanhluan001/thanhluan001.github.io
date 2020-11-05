---
title: "Install Cuda 11 in Ubuntu"
date: 2020-11-05T00:49:14+01:00
draft: false
authors: ["luanpham"]
description: Tutorial to install Cuda for Deep Learning
tags:
    - Installation    
    - Cuda
    - "Deep Learning"
categories:
    - Tutorial
---

Courtesy of [this link](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130) for the information. This other [link](https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu) is an older answer.

To remove most all trace of cuda installations in your system

```bash
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
```

Get the reposistory 

```bash
sudo add-apt-repository ppa:graphics-drivers
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

Install Cuda, no need to install libcudnn (unsure why, can be that the libcudnn is included in new .deb package)

```bash
sudo apt update
sudo apt install cuda
```

As of this writing, it is Cuda 11.1. You can specify the version number like `cuda-10-1` for Cuda 10.1. Check the [repo](http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/) for update-to-date packages

Check for you installation for pytorch

```python
import torch
torch.cuda.is_available()
True
```