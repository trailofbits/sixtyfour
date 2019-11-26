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

#ifndef __APPLE__
#define _GNU_SOURCE
#endif

#ifndef __FILE_NAME__
#define __FILE_NAME__ "MULTICORE"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "output.h"
#include <inttypes.h>
#include <string.h>

#include "avx.h"
#include "avx512.h"
#include "sse.h"
#include "multicore.h"
#include "common.h"
#include "workpool.h"

typedef enum found_status_ {
  sec_unknown,
  sec_found,
  sec_notfound,
  sec_tryagain
} found_status_t;

int CPU_LIMIT = 0;

#ifdef __APPLE__
#include <mach/thread_policy.h>
#include <mach/thread_act.h>

// Support thread affinity on MacOS
// taken from: http://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html
// sadly, MacOS doesn't respect these settings
// but we try anyway

typedef struct cpu_set {
  uint32_t    count;
} cpu_set_t;

static inline void
CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

static inline void
CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

static inline int
CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

int pthread_setaffinity_np(pthread_t thread, size_t cpu_size,
                           cpu_set_t *cpu_set)
{
  thread_port_t mach_thread;
  unsigned core = 0;

  for (core = 0; core < 8 * cpu_size; core++) {
    if (CPU_ISSET(core, cpu_set)) break;
  }
  thread_affinity_policy_data_t policy = { core };
  mach_thread = pthread_mach_thread_np(thread);
  thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                    (thread_policy_t)&policy, 1);
  return 0;
}
#endif // __APPLE__

unsigned char VALID_CPUS[MAX_CPU]= {0};

static thread_args_t thread_args[MAX_CPU];

static pthread_mutex_t cond_guard;
static pthread_cond_t thread_is_done;

#define SLEN 4096
void multicore_print_cpu_map(void) {
  char s[SLEN];
  int np = 0;

  for(int i = 0; i < MAX_CPU; i++) {
    if(np > SLEN-2) {
      break;
    }

    if(VALID_CPUS[i] == 1) {
      int v = snprintf(s+np, SLEN-np, "%d,", i);
      if(v < 0) {
        log_error(__FILE_NAME__, "Could not print string!\n");
        exit(1);
      }
      np += v;
    }
  }

  log_output(__FILE_NAME__, "Operating on CPUs: %s\n", s);
}
#undef SLEN

// parse and set a CPU mapping
bool multicore_set_cpu_map(const char *map) {

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu <= 0) {
    perror("Could not get cpu count");
    exit(-1);
  }

  // clear valid CPUs
  memset(VALID_CPUS, 0, sizeof(VALID_CPUS[0]) * MAX_CPU);

  char *s = strdup(map);
  char *rest = s;
  char *tok;

  if (NULL == rest) {
    log_error(__FILE_NAME__, "Could not duplicate string");
    exit(1);
  }

  while ((tok = strtok_r(rest, ",", &rest))) {
    unsigned int cpu = strtoul(tok, NULL, 0);
    if (cpu < MAX_CPU && cpu < ncpu) {
      VALID_CPUS[cpu] = 1;
    } else {
      log_error(__FILE_NAME__, "Tried to set an invalid CPU: %d [%s]\n", cpu, tok);
      free(s);
      return false;
    }
  }

  free(s);
  return true;
}

// figure out the next CPU a worker thread
// should attach to
static int multicore_next_valid_cpu() {
  for(int i = 0; i < MAX_CPU; i++) {
    if(VALID_CPUS[i] == 1) {
      VALID_CPUS[i] = 0;
      return i;
    }
  }
  return -1;
}

bool multicore_check(void) {
  // used to check for ncpu > 1; but it is equally valid
  // to test multicore and threading overhead on one cpu
  return true;
}

// check if one thread found secret
// or if all threads expired without finding it
static void check_if_found(long ncpu, found_status_t *status) {

  int done_count = 0;

  for(int i = 0; i < ncpu; i++ ) {
    // one thread got it
    if(thread_args[i].found) {
      *status = sec_found;
      return;
    }

    if(thread_args[i].done) {
      done_count += 1; }
  }
  
  // all threads done, no secret
  if(ncpu == done_count) {
    *status = sec_notfound;
    return;
  }

  // some threads are still working
  *status = sec_tryagain;
}

static void* search_thread(void *arg) {
  thread_args_t *targ = (thread_args_t*)arg;
  uint64_t nops = 0;
  uint64_t (*method_func)(uint64_t, uint64_t, uint64_t, bool*);

  // make this thread cancellable
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
  // make it cancellable *immediately*
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

  uint64_t start, stop;

  char *method = NULL;
  targ->found = false;

  if(__builtin_cpu_supports("avx512bw")) {
    method = "AVX512";
    method_func = avx512_do_range;
  } else if(__builtin_cpu_supports("avx2")) {
    method = "AVX2";
    method_func = avx2_do_range;
  } else if(__builtin_cpu_supports("sse4.1")) {
    method = "SSE4.1";
    method_func = sse_do_range;
  } else {
    log_error(__FILE_NAME__, "Neither AVX512 nor AVX2 nor SSE4.1 are supported on this CPU. Exiting\n");
    method_func = NULL;
    exit(-1);
  }

  PoolStatus is_done = workpool_get_chunk(&start, &stop);

  while (is_done != PoolFinished) {
    //log_output(__FILE_NAME__, "TID [%" PRIx64 "] doing a workitem\n",(uint64_t)targ->tid);
    nops += method_func(start, stop, targ->secret, &targ->found);

    if(true == targ->found) {
      break;
    }

    if(true == targ->done) {
      // someone else told us we're done
      // bail out early
      targ->nops = nops;
      log_output(__FILE_NAME__, "TID[%" PRIx64"]: Exiting since someone told us we're done\n",
          (uint64_t)targ->tid);
      return NULL;
    }

    do {
      is_done = workpool_get_chunk(&start, &stop);
    } while (is_done == PoolTryAgain);
  }

  targ->done = true;
  targ->nops = nops;

  log_output(__FILE_NAME__, "Thread [%" PRIx64 "] done [%" PRIu64 "] ops using [%s]\n", (uint64_t)targ->tid, nops, method);

  // we found it! signal to main thread to stop everything
  if(true == targ->found ) {
    log_output(__FILE_NAME__, "TID[%" PRIx64"]: Secret [%" PRIx64 "] found between [%" PRIx64 "] - [%" PRIx64 "]\n", 
        (uint64_t)targ->tid,
        targ->secret, start, stop);
    pthread_mutex_lock(&cond_guard);
    pthread_cond_signal(&thread_is_done);
    pthread_mutex_unlock(&cond_guard);
  } else {
    found_status_t f = sec_unknown;
    check_if_found(targ->ncpu, &f);
    switch(f) {
      case sec_found:
        // found it!
        // fall through
      case sec_notfound:
        // we were the last thread, signal anyway
        pthread_mutex_lock(&cond_guard);
        pthread_cond_signal(&thread_is_done);
        pthread_mutex_unlock(&cond_guard);
        break;
      case sec_tryagain:
        break;
      default:
        log_error(__FILE_NAME__, "Unknown value in 'are we done?' loop\n");
        // other threads still running, ignore
        break;
    }
  }

  return NULL;
}

// estimate ops completed by all threads
static uint64_t estimate_nops(long ncpu) {

  uint64_t est_nops = 0;

  int count = 0;

  // use real values for all completed threads
  for(int i = 0; i < ncpu; i++ ) {
    if(thread_args[i].nops > 0) {
      est_nops += thread_args[i].nops;
      count += 1;
    }
  }

  if (count < ncpu) {
    log_output(__FILE_NAME__,
               "WARNING: Not all CPUs completed a work item [%d/%d]\n", count,
               ncpu);
  }

  return est_nops;
}

bool multicore_launch_threads(long ncpu, uint64_t secret, uint64_t h_start, uint64_t h_end) {
  if(ncpu > MAX_CPU) {
    log_output(__FILE_NAME__, "too many CPUs [%ld]. Limiting to [%d]\n",
        ncpu, MAX_CPU);
    ncpu = MAX_CPU;
  } else {
    log_output(__FILE_NAME__, "Found [%ld] enabled processors\n", ncpu);
  }

  // initialize condition variable and mutex used to detect
  // when a thread has found the needle in the haystack
  pthread_mutex_init(&cond_guard, NULL);
  pthread_cond_init(&thread_is_done, NULL);

  if(!workpool_is_set()) {
    workpool_set(h_start, h_end, ncpu);
  }

  // assign each thread a range to search
  for(uint64_t i = 0; i < (uint64_t)ncpu; i++) {
    // clean up this thread's area
    memset(&thread_args[i], 0, sizeof(thread_args_t));

    thread_args[i].secret =   secret;
    thread_args[i].done   =   false;
    thread_args[i].ncpu   =   ncpu;

  }


  // 3) start threads
  //
  // lock notification condition so we know when a thread finds secret
  pthread_mutex_lock(&cond_guard);

  // initialize each thread
  int i;
  log_output(__FILE_NAME__, "Starting threads\n");
  for(i = 0; i < ncpu; i++) {
    int ret;
    
    // thread attributes used to specify a cpu affinity
    pthread_attr_t attr;
    ret = pthread_attr_init(&attr);

    // bind each thread to a specific CPU
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    int valid_cpu = multicore_next_valid_cpu();
    if(valid_cpu < 0) {
      log_error(__FILE_NAME__, "Ran out of CPUs!\n");
      exit(-1);
    }
    log_output(__FILE_NAME__, "Thread will use CPU: %d\n", valid_cpu);
    CPU_SET(valid_cpu, &cpus);

    // create the thread
    log_output(__FILE_NAME__, ".\n");
    ret = pthread_create(&(thread_args[i].tid), &attr, search_thread, &(thread_args[i]) );
    if(ret < 0) {
      perror("Could not create thread");
      return false;
    }

    ret = pthread_setaffinity_np(thread_args[i].tid, sizeof(cpu_set_t), &cpus);
    if(ret < 0) {
      perror("Could not set thread affinity");
      return false;
    }
    
  }
  log_output(__FILE_NAME__, "\n");
  return true;

}

void multicore_init_cpus(int count) {
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if(ncpu <= 0) {
      perror("Could not get cpu count");
      exit(-1);
  }

  if(ncpu > MAX_CPU) {
    ncpu = MAX_CPU;
  }

  if(count > 0 && ncpu > count) {
    ncpu = count;
  }

  memset(VALID_CPUS, 0, sizeof(VALID_CPUS));
  for(int i = 0; i < ncpu; i++) {
    VALID_CPUS[i] = 1;
  }
}

long multicore_get_ncpus(void) {
  long sum = 0;
  for(int i = 0; i < MAX_CPU; i++) {
    if(VALID_CPUS[i] == 1) {
      sum += 1;
    }
  }

  return sum;
}

void multicore_wait_for_threads(long ncpu, method_results_t *results) {

  results->found = false;
  results->ops_done = 0;
  bool found = false;

  found_status_t status = sec_unknown;
  bool threads_done = false;
  while(false == threads_done) {
    pthread_cond_wait(&thread_is_done, &cond_guard);
    check_if_found(ncpu, &status);
    switch(status) {
      case sec_found: // found the secret
        found = true;
        threads_done = true;
        break;
      case sec_notfound: // secret not found by any threads
        found = false;
        threads_done = true;
        break;
      case sec_tryagain: // someone is still working
        // redundant, here for clarity
        threads_done = false;
        break;
      default:
        log_output(__FILE_NAME__, "Unknown failure in status check loop\n");
        exit(-1);
    }
  }
  pthread_mutex_unlock(&cond_guard);

  // wait for threads to complete work items
  // tell them not to fetch anymore
  log_output(__FILE_NAME__, "Finished. Stopping threads...\n");
  for(int i = 0; i < ncpu; i++) {
      thread_args[i].done = true;
  }

  log_output(__FILE_NAME__, "Waiting for threads to finish last work items\n");
  for(int i = 0; i < ncpu; i++) {
    void *v;
    pthread_join(thread_args[i].tid, &v);
  }

  uint64_t nops = estimate_nops(ncpu);
  
  results->ops_done = nops;
  results->found = found;
}

uint64_t multicore_method(uint64_t secret, bool *found, uint64_t h_start, uint64_t h_end) {
  // 1) Get CPU count
  long ncpu = multicore_get_ncpus();
  method_results_t res = {0};
  
  if(false == multicore_launch_threads(ncpu, secret, h_start, h_end)) {
    log_error(__FILE_NAME__, "Could not launch worker threads!");
    exit(-1);
  }

  if(0 == ncpu) {
    log_error(__FILE_NAME__, "No CPUs available; aborting method\n");
    res.found = false;
    res.ops_done = 0;
  } else {
    // while loop checking that something was found
    // and which thread found it
    log_output(__FILE_NAME__, "Waiting for worker threads\n");
    multicore_wait_for_threads(ncpu, &res);
  }

  *found = res.found;
  return res.ops_done;
}

