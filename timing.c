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

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

#include "timing.h"
#include "output.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "TIMING"
#endif

static void read_clock(struct timespec *tm) {
//  if(0 != clock_gettime(CLOCK_PROCESS_CPUTIME_ID, tm)) {
  if(0 != clock_gettime(CLOCK_REALTIME, tm)) {
    perror("Could not read clock: " );
    exit(-1);
  }
}

void start_timer(elapsed_time_t *timer) {

  memset(&timer->start_time, 0, sizeof(struct timespec));
  memset(&timer->stop_time, 0, sizeof(struct timespec));

  read_clock(&timer->start_time);
}

void stop_timer(elapsed_time_t *timer) {
  read_clock(&timer->stop_time);
}

static int time_diff(struct timespec *result, struct timespec *x, struct timespec *y)
{
  const unsigned long long nsec_max = 1000000000ULL;
  if(y->tv_nsec > x->tv_nsec) {
    x->tv_sec -= 1;
    x->tv_nsec += nsec_max;
  } 

  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  return 0;
}

static void print_time(const struct timespec *tm, uint64_t nops, const char *mname) {
  unsigned long long hours = tm->tv_sec / 3600;
  unsigned long long rem = tm->tv_sec - (hours * 3600);
  unsigned long long minutes = rem / 60;
  rem -= minutes * 60;
  unsigned long long seconds = rem;

  long msec = tm->tv_nsec / 1000000;
  
  log_output(__FILE_NAME__, "Method: %s\n", mname);
  log_output(__FILE_NAME__, "Elapsed Time: %02lld:%02lld:%02lld.%03ld\n",
        hours,
        minutes,
        seconds,
        msec);

  long double full_msec = msec + (1000 * tm->tv_sec);
  if(full_msec <= 0.0) {
    log_error(__FILE_NAME__, "Not enough time spent to measure, aborting\n");
    exit(1);
  }
  long double opsperms = nops / full_msec;

  log_output(__FILE_NAME__, "Estimated ops per:\n"
         "    millisecond: %20.0Lf\n"
         "         second: %20.0Lf\n"
         "         minute: %20.0Lf\n"
         "           hour: %20.0Lf\n"
         "            day: %20.0Lf\n" ,
         opsperms,
         opsperms * 1000,
         opsperms * 1000 * 60,
         opsperms * 1000 * 60 * 60,
         opsperms * 1000 * 60 * 60 * 24
         );

  // how many more times do we  need?

  uint64_t msec_intervals_needed = UINT64_MAX / (uint64_t)opsperms;
  uint64_t full_space_secs = msec_intervals_needed / 1000;
  
  unsigned int fs_years = full_space_secs / (60 * 60 * 24 * 365);
  full_space_secs = full_space_secs % (60 * 60 * 24 * 365);
  unsigned int fs_days = full_space_secs / (60 * 60 * 24);
  full_space_secs = full_space_secs % (60 * 60 * 24);
  unsigned int fs_hours = full_space_secs / (60*60);
  full_space_secs = full_space_secs % (60 * 60);
  unsigned int fs_minutes = full_space_secs / 60;
  full_space_secs = full_space_secs % 60;


  char time_string[128];
  snprintf(time_string, sizeof(time_string), "%04dY %03dD %02dH %02dM %02dS",
      fs_years,
      fs_days,
      fs_hours,
      fs_minutes,
      (unsigned int)full_space_secs);

  log_output(__FILE_NAME__, "Time to search all 64-bits: %s\n", time_string);

  printf("{\n");
  printf("  \"finishwhen\": \"%s\",\n", time_string);
  printf("  \"method\": \"%s\",\n", mname);
  printf("  \"etime\": \"%02lld:%02lld:%02lld.%03ld\",\n",
        hours,
        minutes,
        seconds,
        msec);
  printf("  \"ms\":  \"%Lf\",\n",   opsperms);
  printf("  \"sec\": \"%Lf\",\n",   opsperms * 1000);
  printf("  \"min\": \"%Lf\",\n",   opsperms * 1000 * 60);
  printf("  \"hour\":\"%Lf\",\n",   opsperms * 1000 * 60 * 60);
  printf("  \"day\": \"%Lf\"\n",    opsperms * 1000 * 60 * 60 * 24);
  printf("}\n");

}

void show_elapsed_time(elapsed_time_t *timer, uint64_t nops, const char *mname) {
  struct timespec diff;

  time_diff(&diff, &timer->stop_time, &timer->start_time);

  print_time(&diff, nops, mname);
} 

uint64_t get_ops_per_ms(elapsed_time_t *timer, uint64_t nops) {
  struct timespec diff;

  time_diff(&diff, &timer->stop_time, &timer->start_time);
  long msec = diff.tv_nsec / 1000000;
  msec += 1000 * diff.tv_sec;

  uint64_t opsperms = nops / (uint64_t)msec;

  return opsperms;
}
