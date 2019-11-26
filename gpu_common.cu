/*
 * Copyright (c) 2019 Trail of Bits, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "output.h"
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include "gpu_common.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "GPU_COMMON"
#endif

device_info_t CUDA_DEVICES[MAX_CUDA_DEVICES] = {0};
int avilable_gpus = 0;

extern "C" {
  int GPU_LIMIT = 0;
}

// figure out how many GPUs we have
// and how many threads/blocks to use for
// each GPU
extern "C" void gpu_get_device_info(void) {

  int devices = 0;
  cudaGetDeviceCount(&devices);

  if(devices > MAX_CUDA_DEVICES) {

    log_output(__FILE_NAME__, "Found more than %d devices, only using %d\n", MAX_CUDA_DEVICES, MAX_CUDA_DEVICES);
    devices = MAX_CUDA_DEVICES;
  } else {
    log_output(__FILE_NAME__, "Found %d devices\n", devices);
  }

  if(GPU_LIMIT > 0 && devices > GPU_LIMIT) {
    log_output(__FILE_NAME__, "Artificially limiting to %d of %d GPUs\n", GPU_LIMIT, devices);
    avilable_gpus = GPU_LIMIT;
  } else{
    avilable_gpus = devices;
  }


  for (int i = 0; i < avilable_gpus; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    log_output(__FILE_NAME__, "GPU[%d]: Device Number: %d\n", i, i);
    log_output(__FILE_NAME__, "GPU[%d]:   Device name: %s\n", i, prop.name);
    log_output(__FILE_NAME__, "GPU[%d]:   Clock Rate (KHz): %d\n", i,
           prop.clockRate);

    cudaSetDevice(i);

    int blocks;
    int threads;

    // find a decent threads/block combo
    if(cudaSuccess != cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, findit, 0, 0)) {
      log_output(__FILE_NAME__, "GPU[%d]: Could not find max occupancy for GPU\n", i);
      exit(-1);
    }

    log_output(__FILE_NAME__, "GPU[%d]:   Blocks: %d\n", i, blocks);
    log_output(__FILE_NAME__, "GPU[%d]:   Threads: %d\n", i, threads);

    CUDA_DEVICES[i].blocks = blocks;
    CUDA_DEVICES[i].threads = threads;
    CUDA_DEVICES[i].clockrate = prop.clockRate;
    CUDA_DEVICES[i].id = i;
    CUDA_DEVICES[i].finished = false;
  }
}

// how many "work pieces" is the input space
// being divided into?
extern "C" int gpu_get_workers(void) {
  int total_size = 0;
  for(int i = 0; i < avilable_gpus; i++) {
    int my_blocks = CUDA_DEVICES[i].blocks;
    int my_threads = CUDA_DEVICES[i].threads;
    int my_alloc =  my_blocks * my_threads;
    total_size += my_alloc;
  }

  return total_size;
}
