/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_UTIL_NOTIFICATION_H_
#define TENSORFLOW_UTIL_NOTIFICATION_H_

#include <assert.h>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <iostream>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {}

  void Notify() {
    mutex_lock l(mu_);
    assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  bool HasBeenNotified() {
    mutex_lock l(mu_);
    return notified_;
  }

  void WaitForNotification() {
    mutex_lock l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

  bool WaitForNotificationWithTimeout(int64 timeout_in_ms) {
    mutex_lock l(mu_);
    std::cv_status s =
        cv_.wait_for(l, std::chrono::milliseconds(timeout_in_ms));
    return (s == std::cv_status::timeout) ? true : false;
  }

 private:
  mutex mu_;
  condition_variable cv_;
  bool notified_;
};

class MultiUseNotification {
 public:
  MultiUseNotification() : 
      times_notified_(0), completed_(false) {}
  ~MultiUseNotification() {}

  void Notify() {
    mutex_lock l(mu_);
    // assert(!notified_);
    times_notified_ ++;
    cv_.notify_all();
  }

  bool TimesNotified() {
    mutex_lock l(mu_);
    return times_notified_;
  }

  void WaitForNotification() {
    mutex_lock l(mu_);
    int prev_times_notified = times_notified_;
    std::cout << "--- Waiting for notification: prev_times_notified_ = "
              << prev_times_notified << std::endl;
    while (times_notified_ == prev_times_notified) {
      cv_.wait(l);
    }
  }

  void MarkAsCompleted() {
    mutex_lock l(mu_);
    completed_ = true;
  }

  bool IsCompleted() {
    mutex_lock l(mu_);
    return completed_;
  }

 private:
  mutex mu_;
  condition_variable cv_;
  int times_notified_;
  bool completed_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_NOTIFICATION_H_
