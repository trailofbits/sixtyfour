#!/bin/bash
set -e
set +x

sudo apt-get update
sudo apt-get install -yq git wget cmake ninja-build clang-7 screen

if [ "$(which nvidia-smi)" != "" ]
then
  CUDA_BUILD=True
else
  CUDA_BUILD=False
fi

mkdir -p /tmp/cmake
pushd /tmp/cmake
wget https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4-Linux-x86_64.sh
chmod +x ./cmake-3.15.4-Linux-x86_64.sh
sudo ./cmake-3.15.4-Linux-x86_64.sh --skip-license --prefix=/usr/local
popd

mkdir -p build && pushd build
CC=clang-7 /usr/local/bin/cmake -G Ninja -DCUDA_BUILD=${CUDA_BUILD} ..
cmake --build .
popd

screen /bin/bash -c scripts/measure.sh
