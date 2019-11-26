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
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include "gpu.h"
#include "gpu_common.h"
#include <unistd.h>
#include "workpool.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "GPU"
#endif

#ifdef __cplusplus
extern "C"
#endif
bool gpu_check(void) {
  return true;
}

__device__ bool global_found = false;

// the main GPU kernel!
// do a simple while loop over our workslice
__global__ void findit(uint64_t secret, entry_t *args) {
  if(true == global_found) {
    return;
  }
  int index = blockIdx.x * blockDim.x + threadIdx.x ;

  int found = 0;

  uint64_t i = 0;
  uint64_t stop = args[index].stop;
  uint64_t start = args[index].start;

  i = start;
  do {
    if(secret == i) {
      found = 1;
    }
    ++i;
  }while(i < stop);


  args[index].nops = i - start;

  if(found) {
    global_found = true;
    args[index].which = secret;
    //printf(__FILE_NAME__ ": Found secret at index=%d\n", index);
  }
}

#ifdef __cplusplus
// NVCC can build as C++
extern "C"
#endif

static uint64_t gpu_sum_nops(entry_t *ha, int devIdx, bool *found) {
  uint64_t nops = 0;
  int slicey = CUDA_DEVICES[devIdx].slice_index;
  int sizey = CUDA_DEVICES[devIdx].threads * CUDA_DEVICES[devIdx].blocks;

  log_output(__FILE_NAME__,
             "GPU[%d]: Saving results for slices [%06d] - [%06d]\n", devIdx,
             slicey, slicey + sizey - 1);

  // write results to big host array
  cudaMemcpy(&ha[slicey], CUDA_DEVICES[devIdx].da,
             sizeof(entry_t) * sizey, cudaMemcpyDeviceToHost);

  // update number of operations performed
  for (int i = slicey; i < slicey + sizey; i++) {
    nops += ha[i].nops;
    if(ha[i].which != 0) {
      log_output(__FILE_NAME__, "GPU: Secret found by thread [%" PRIu64 "]: [%" PRIx64 "] < [%" PRIx64 "] < [%" PRIx64 "]\n",
          i, ha[i].start, ha[i].which, ha[i].stop);
      *found = true;
    }
  }

  return nops;
}

#ifdef __cplusplus
// NVCC can build as C++
extern "C"
#endif

static int gpu_fill_workslice(entry_t *ha, int devIdx) {
    int my_blocks = CUDA_DEVICES[devIdx].blocks;
    int my_threads = CUDA_DEVICES[devIdx].threads;
    int my_alloc =  my_blocks * my_threads;

    int requests_filled = 0;

    log_output(__FILE_NAME__, "GPU[%d] Filling work slices [%06d - %06d]\n", 
      devIdx,
      CUDA_DEVICES[devIdx].slice_index,
      CUDA_DEVICES[devIdx].slice_index + my_alloc - 1);

    int slice = CUDA_DEVICES[devIdx].slice_index;
    int limit = slice+my_alloc;
    // how many slices this GPU is allocated
    // the starting slice; used to write results back into main array
    for (; slice < limit; slice++) {
      uint64_t start, end;
      PoolStatus p = workpool_get_chunk(&start, &end);
      while (p == PoolTryAgain) {
        p = workpool_get_chunk(&start, &end);
      }
      if (PoolFinished == p) {
        break;
      } else {
        requests_filled++;
      }

      // tell it about start, stop, secret
      ha[slice].start = start;
      ha[slice].stop = end;
      ha[slice].nops = 0;
      ha[slice].which = 0;
    }
    // zero out what is left
    for (; slice < limit; slice++) {
      memset(&ha[slice], 0, sizeof(*ha));
    }

    return requests_filled;
}

bool gpu_launch_kernel(kernel_info_t *info, uint64_t secret, uint64_t h_start, uint64_t h_end) {
  int total_blocks = 0;
  int total_threads = 0;
  uint64_t total_size = 0;

  // how many independent processing units do we have to work with?
  for(int i = 0; i < avilable_gpus; i++) {
    int my_blocks = CUDA_DEVICES[i].blocks;
    int my_threads = CUDA_DEVICES[i].threads;
    // how many slices this GPU is allocated
    int my_alloc =  my_blocks * my_threads;
    CUDA_DEVICES[i].slice_index = total_size;

    total_blocks += my_blocks;
    total_threads += my_threads;
    total_size += my_alloc;
  }

  info->total_size = total_size;

  // set up work pool
  if(!workpool_is_set()) {
    workpool_set(h_start, h_end, info->total_size);
  }

  log_output(__FILE_NAME__, "GPU: Using %d blocks of %d threads over %d GPUs\n", total_blocks, total_threads, avilable_gpus);

  //entry_t *hostArgs = (entry_t*)malloc(total_size * sizeof(entry_t));
  entry_t *hostArgs = NULL;
  cudaMallocHost(&hostArgs, total_size * sizeof(entry_t));
  if(hostArgs == NULL) {
    log_output(__FILE_NAME__, "GPU: Could not allocate %lx bytes of memory\n", total_size *sizeof(entry_t));
    return false;
  }
  info->hostArgs = hostArgs;

  // erase global "Found" flag
  cudaMemset(&global_found, 0, sizeof(global_found));

  // start a kernel on each device doing its own part of the workload
  for(int i = 0; i < avilable_gpus; i++) {
    int my_blocks = CUDA_DEVICES[i].blocks;
    int my_threads = CUDA_DEVICES[i].threads;
    // how many slices this GPU is allocated
    int my_alloc =  my_blocks * my_threads;
    // the starting slice; used to write results back into main array
    int slice_idx = CUDA_DEVICES[i].slice_index;

    gpu_fill_workslice(info->hostArgs, i);

    cudaSetDevice(i);

    // allocate some workspace in the GPU
    if(cudaSuccess != 
        cudaMalloc(&(CUDA_DEVICES[i].da), my_alloc * sizeof(entry_t))) {
      log_error(__FILE_NAME__, "GPU[%d]: Could not allocate %ld bytes\n", i, my_alloc * sizeof(entry_t));
      return false;
    }

    // create a stream
    if(cudaSuccess != cudaStreamCreate(&CUDA_DEVICES[i].stream)) {
      log_error(__FILE_NAME__, "GPU[%d]: Could not create steam\n", i);
      return false;
    }
    // write workload into to gpu, asynchronously
    cudaMemcpyAsync(CUDA_DEVICES[i].da, &hostArgs[slice_idx], my_alloc * sizeof(entry_t),
               cudaMemcpyHostToDevice, 
               CUDA_DEVICES[i].stream);

  }

  for(int i = 0; i < avilable_gpus; i++) {
    cudaSetDevice(i);
    // launch kernel on stream and set it to work
    // this call is asynchronous!
    findit<<<
      CUDA_DEVICES[i].blocks,
      CUDA_DEVICES[i].threads,
      0,
      CUDA_DEVICES[i].stream>>>
      (secret, CUDA_DEVICES[i].da);
  }

  return true;
}

#ifdef __cplusplus
// NVCC can build as C++
extern "C"
#endif
void gpu_wait_for_streams(const struct kernel_info_t *ki, uint64_t secret, method_results_t *results) {
  int done_streams = 0;
  bool found = false;

  uint64_t total_nops = 0;

  results->found = false;
  results->ops_done = 0;

  // wait for streams to end (via polling)
  entry_t *hostArgs = ki->hostArgs;
  while(done_streams < avilable_gpus) {
    for(int i = 0; i < avilable_gpus; i++) {
      // some devices already finished... don't check them again
      if(CUDA_DEVICES[i].finished == false) {
        // is the stream done?
        if(cudaSuccess == cudaStreamQuery(CUDA_DEVICES[i].stream)) {
          // yes, great!
          log_output(__FILE_NAME__, "GPU[%d]: Operations complete\n", i);
          cudaSetDevice(i);

          // fetch results from this workload
          // add to total ops, and see if 
          // the secret was found
          total_nops += gpu_sum_nops(hostArgs, i, &found);

          // if secret was found, or there is no more work to give
          // then exit the stream
          if(found || 0 == gpu_fill_workslice(hostArgs, i) ) {
            // done with the stream
            cudaFree(CUDA_DEVICES[i].da);
            cudaStreamDestroy(CUDA_DEVICES[i].stream);
            CUDA_DEVICES[i].finished = true;
            // one step closer to completion
            ++done_streams;
          } else {
            int slicey = CUDA_DEVICES[i].slice_index;
            int b = CUDA_DEVICES[i].blocks;
            int d = CUDA_DEVICES[i].threads;
            int sizey = b*d;
            // work to do still! copy stuff over
            // the hostArgs should have been
            // re-populated by gpu_fill_workslice
            cudaMemcpyAsync(CUDA_DEVICES[i].da, &hostArgs[slicey], sizey * sizeof(entry_t),
                      cudaMemcpyHostToDevice, 
                      CUDA_DEVICES[i].stream);
            // re-launch the kernel
            findit<<<b, d, 0, CUDA_DEVICES[i].stream>>>(secret, CUDA_DEVICES[i].da);
          }
          
        }
      }
    }
    // wait for 1ms until polling again
    usleep(1000);
  }

  // all streams done.
  // reset each device
  for(int i = 0; i < avilable_gpus; i++) {
    cudaSetDevice(i);
    cudaDeviceReset();
  }

  log_output(__FILE_NAME__, "Performed: [%" PRIu64 "] operations\n", total_nops);

  results->found = found;
  results->ops_done = total_nops;
}

#ifdef __cplusplus
// NVCC can build as C++
extern "C"
#endif
uint64_t gpu_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {

  kernel_info_t ki;
  if(false == gpu_launch_kernel(&ki, secret, h_start, h_end)) {
    log_error(__FILE_NAME__, "Failed to launch kernel\n");
    exit(-1);
  }

  method_results_t res = {0};
  gpu_wait_for_streams(&ki, secret, &res);

  if (ki.hostArgs != NULL) {
    cudaFreeHost(ki.hostArgs);
  }
  *found = res.found;
  return res.ops_done;
}
