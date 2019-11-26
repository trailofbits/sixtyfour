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

#include "gpu.h"
#include "gpu_common.h"
#include "multicore.h"
#include "timing.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "output.h"
#include "workpool.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "GPU_MULTICORE"
#endif

static void *gpu_wait_thread(void *arg);
static void *cpu_wait_thread(void *arg);

#define THREAD_COUNT 2

static thread_args_t gpu_args = {0};
static thread_args_t cpu_args = {0};

typedef void* (*thread_func_t)(void*);

static thread_args_t *worker_threads[THREAD_COUNT] = {&gpu_args, &cpu_args};
static thread_func_t thread_funcs[THREAD_COUNT] = {gpu_wait_thread, cpu_wait_thread};
static const char *thread_names[THREAD_COUNT] = {"GPU", "CPU"};

static pthread_mutex_t cond_guard;
static pthread_cond_t thread_is_done;

#ifdef __cplusplus
extern "C"
#endif
bool gpu_multicore_check(void) {
  return multicore_check() && gpu_check();
}

static void *gpu_wait_thread(void *arg) {
  // make this thread cancellable
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
  // make it cancellable *immediately*
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
  if (NULL == arg) {
    log_error(__FILE_NAME__, "GPU Thread given a bad argument\n");
    exit(-1);
  }

  thread_args_t *ti = (thread_args_t *)(arg);

  // just call the method runner directly with limited arguments
  // TODO(artem): Control logging :)
  ti->nops = gpu_method(ti->secret, &(ti->found), ti->start, ti->stop);
  ti->done = true;
  pthread_mutex_lock(&cond_guard);
  pthread_cond_signal(&thread_is_done);
  pthread_mutex_unlock(&cond_guard);
  return NULL;
}

static void *cpu_wait_thread(void *arg) {
  // make this thread cancellable
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
  // make it cancellable *immediately*
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
  if (NULL == arg) {
    log_error(__FILE_NAME__, "CPU Thread given a bad argument\n");
    exit(-1);
  }

  thread_args_t *ti = (thread_args_t *)(arg);

  // just call the method runner directly with limited arguments
  ti->nops = multicore_method(ti->secret, &(ti->found), ti->start, ti->stop);

  ti->done = true;
  pthread_mutex_lock(&cond_guard);
  pthread_cond_signal(&thread_is_done);
  pthread_mutex_unlock(&cond_guard);

  return NULL;
}

#ifdef __cplusplus
extern "C"
#endif
uint64_t gpu_multicore_method(uint64_t secret, bool *found, uint64_t h_start,
                              uint64_t h_end) {
  *found = false;

  // initialize notification mutex
  pthread_mutex_init(&cond_guard, NULL);
  pthread_cond_init(&thread_is_done, NULL);

  for(int i = 0; i < THREAD_COUNT; i++) {
    memset(worker_threads[i], 0, sizeof(thread_args_t));
    worker_threads[i]->done = false;
    worker_threads[i]->found = false;
    // these values are ignored since its fetched from a workpool
    worker_threads[i]->start = h_start;
    worker_threads[i]->stop = h_end;
    worker_threads[i]->secret = secret;
  }

  int gpu_workers = gpu_get_workers();
  int cpu_workers = multicore_get_ncpus();
  int total_workers = cpu_workers + gpu_workers;

  log_output(__FILE_NAME__, "Total workers = [%d] [CPU: %d][GPU: %d]\n",
    total_workers, cpu_workers, gpu_workers);

  // Set up work items each thread will pick
  if (!workpool_is_set()) {
    workpool_set(h_start, h_end, total_workers);
  }

  log_output(__FILE_NAME__, "Creating worker threads\n");
  for (int i = 0; i < 2; i++) {
    pthread_attr_t attr;
    int ret = pthread_attr_init(&attr);
    int rv = pthread_create(&(worker_threads[i]->tid), &attr, thread_funcs[i],
                            worker_threads[i]);
    if(ret < 0) {
      perror("Could not create thread");
      exit(-1);
    }
    log_output(__FILE_NAME__, ".\n");
  }

  log_output(__FILE_NAME__, "Waiting on a method to finish...\n");

  // merge multicore wait loop with gpu wait loop
  bool threads_done = false;
  while (false == threads_done) {
    pthread_cond_wait(&thread_is_done, &cond_guard);
    log_output(__FILE_NAME__, "A thread finished!\n");
    int done_count = 0;
    for (int i = 0; i < THREAD_COUNT; i++) {
      // count how many threads are done
      if (true == worker_threads[i]->done) {
        done_count += 1;
      }

      // only output the found message once
      if (true == worker_threads[i]->found && false == *found) {
        log_output(__FILE_NAME__, "[%s] found the secret!\n",
               thread_names[i]);
        *found = true;
        log_output(__FILE_NAME__, "Waiting for other threads to finish to get "
               "accurate performance stats\n");
      }

      // stop loop when all threads end
      if (THREAD_COUNT == done_count) {
        threads_done = true;
      }
    }
  }
  log_output(__FILE_NAME__, "All threads done!\n");

  pthread_mutex_unlock(&cond_guard);

  // just to be sure, in case we ever bail out early
  for (int i = 0; i < THREAD_COUNT; i++) {
    void *p;
    pthread_join(worker_threads[i]->tid, &p);
  }

  // calculate a combined nops from both methods
  uint64_t total_nops = 0;
  for(int i = 0; i < THREAD_COUNT; i++) {
      total_nops += worker_threads[i]->nops;
  }
  log_output(__FILE_NAME__, "There were [%016" PRIu64 "] operations performed\n",
        total_nops);

  // get percentage performed by each method
  for (int i = 0; i < THREAD_COUNT; i++) {
    uint64_t my_nops = worker_threads[i]->nops;
    log_output(__FILE_NAME__, "\t[%s] did [%016" PRIu64 "] Operations, or [%02.02lf] percent\n",
           thread_names[i], my_nops,
           (double)my_nops * 100.0 / (double)total_nops);
  }

  return total_nops;
}
