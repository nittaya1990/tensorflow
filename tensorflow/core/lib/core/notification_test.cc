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

#include <unistd.h>

#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

const int microsec_to_sleep = 10 * 1000;  // 10 ms

TEST(NotificationTest, TestSingleNotification) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  int counter = 0;
  Notification start;
  Notification proceed;
  thread_pool->Schedule([&start, &proceed, &counter] {
    start.Notify();
    proceed.WaitForNotification();
    ++counter;
  });

  // Wait for the thread to start
  start.WaitForNotification();

  // The thread should be waiting for the 'proceed' notification.
  EXPECT_EQ(0, counter);

  // Unblock the thread
  proceed.Notify();

  delete thread_pool;  // Wait for closure to finish.

  // Verify the counter has been incremented
  EXPECT_EQ(1, counter);
}

TEST(NotificationTest, TestMultipleThreadsWaitingOnNotification) {
  const int num_closures = 4;
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", num_closures);

  mutex lock;
  int counter = 0;
  Notification n;

  for (int i = 0; i < num_closures; ++i) {
    thread_pool->Schedule([&n, &lock, &counter] {
      n.WaitForNotification();
      mutex_lock l(lock);
      ++counter;
    });
  }
  Env::Default()->SleepForMicroseconds(microsec_to_sleep);

  EXPECT_EQ(0, counter);

  n.Notify();
  delete thread_pool;  // Wait for all closures to finish.
  EXPECT_EQ(4, counter);
}

TEST(MultiUseNotificationTest, TestNotifyOnce) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  int counter = 0;
  Notification start;
  MultiUseNotification proceed;
  thread_pool->Schedule([&start, &proceed, &counter] {
    start.Notify();

    // Wait for notifications until the object as marked as completed
    while (!proceed.IsCompleted()) {
      proceed.WaitForNotification();
      if (!proceed.IsCompleted()) {
        ++counter;
      }
    }

  });

  // Wait for the thread to start
  start.WaitForNotification();

  // The thread should be waiting for the 'proceed' notification.
  EXPECT_EQ(0, counter);

  // Unblock the thread for the 1st time
  proceed.NotifyOnce();

  // Sleep for a little while for the counter to update
  Env::Default()->SleepForMicroseconds(microsec_to_sleep);

  // Verify the counter has been incremented
  EXPECT_EQ(1, counter);

  // Unblock the thread for the 2nd time
  proceed.NotifyOnce();
  Env::Default()->SleepForMicroseconds(microsec_to_sleep);

  // Verify the counter has been incremented
  EXPECT_EQ(2, counter);

  proceed.MarkAsCompleted();

  delete thread_pool;  // Wait for closure to finish.
  EXPECT_EQ(2, counter);
}

TEST(MultiUseNotificationTest, TestNotifyMultipleTimes) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  int counter = 0;
  Notification start;
  MultiUseNotification proceed;
  thread_pool->Schedule([&start, &proceed, &counter] {
    start.Notify();

    // Wait for notifications until the object as marked as completed
    while (!proceed.IsCompleted()) {
      proceed.WaitForNotification();
      if (!proceed.IsCompleted()) {
        ++counter;
      }
    }

  });

  // Wait for the thread to start
  start.WaitForNotification();

  // The thread should be waiting for the 'proceed' notification.
  EXPECT_EQ(0, counter);

  const int times_to_notify = 42;

  // Unblock the thread for the 1st time
  proceed.Notify(times_to_notify);
  
  // Sleep for a little while for the counter to update
  Env::Default()->SleepForMicroseconds(microsec_to_sleep);

  // Verify the counter has been incremented
  EXPECT_EQ(times_to_notify, counter);

  proceed.MarkAsCompleted();

  delete thread_pool;  // Wait for closure to finish.
  EXPECT_EQ(times_to_notify, counter);
}

}  // namespace
}  // namespace tensorflow
