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

#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>
#include "output.h"

#ifndef __FILE_NAME__
#define __FILE_NAME__ "OUTPUT"
#endif

static pthread_mutex_t output_mutex;

logger_t log_output = log_null;

void initialize_output(void) {
  pthread_mutex_init(&output_mutex, NULL);
}

int log_null (const char *component, const char *fmt, ...)
{
    (void)component;
    (void)fmt;
    return 0;
}

// basic debug logging with a mutex
int log_stdout (const char *component, const char *fmt, ...)
{
    int printed;
    va_list ap;
    va_start (ap, fmt);
    pthread_mutex_lock(&output_mutex);
    printed = fprintf(stdout, "%s: ", component);
    printed += vfprintf(stdout, fmt, ap);
    pthread_mutex_unlock(&output_mutex);
    va_end (ap);
    return printed;
}

int log_error (const char *component, const char *fmt, ...)
{
    int printed;
    va_list ap;
    va_start (ap, fmt);
    pthread_mutex_lock(&output_mutex);
    printed = fprintf(stdout, "%s: ", component);
    printed += vfprintf(stdout, fmt, ap);
    pthread_mutex_unlock(&output_mutex);
    va_end (ap);
    return printed;
}