# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed training and evaluation of a tf-learn/wide&deep model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import urllib

# import numpy as np  # TODO(cais): Remove if not used
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config


# Define command-line flags
flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/census-data",
                    "Directory for storing the cesnsus data data")
flags.DEFINE_string("model_dir", "/tmp/census_wide_and_deep_model",
                    "Directory for storing the model")
flags.DEFINE_string("master_grpc_url", "",
                    "URL to master GRPC tensorflow server, e.g.,"
                    "grpc://127.0.0.1:2222")
flags.DEFINE_integer("num_parameter_servers", 0,
                     "Number of parameter servers")
flags.DEFINE_integer("worker_index", 0,
                     "Worker index (>=0)")
flags.DEFINE_integer("train_steps", 1000, "Number of training steps")
flags.DEFINE_integer("eval_steps", 1, "Number of evaluation steps")

FLAGS = flags.FLAGS


# Constants: Data URLs
TRAIN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Define features for the model
#   1. Categorical base columns.
gender = tf.contrib.layers.sparse_column_with_keys(
    column_name="gender", keys=["female", "male"])
race = tf.contrib.layers.sparse_column_with_keys(
    column_name="race",
    keys=["Amer-Indian-Eskimo",
          "Asian-Pac-Islander",
          "Black",
          "Other",
          "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(
    "marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
    "relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
    "workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

#   2. Continuous base columns.
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")


wide_columns = [
    gender, native_country, education, occupation, workclass,
    marital_status, relationship, age_buckets,
    tf.contrib.layers.crossed_column([education, occupation],
                                     hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([native_country, occupation],
                                     hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([age_buckets, race, occupation],
                                     hash_bucket_size=int(1e6))]

deep_columns = [
    tf.contrib.layers.embedding_column(workclass, dimension=8),
    tf.contrib.layers.embedding_column(education, dimension=8),
    tf.contrib.layers.embedding_column(marital_status, dimension=8),
    tf.contrib.layers.embedding_column(gender, dimension=8),
    tf.contrib.layers.embedding_column(relationship, dimension=8),
    tf.contrib.layers.embedding_column(race, dimension=8),
    tf.contrib.layers.embedding_column(native_country, dimension=8),
    tf.contrib.layers.embedding_column(occupation, dimension=8),
    age, education_num, capital_gain, capital_loss, hours_per_week]

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status",
                       "occupation", "relationship", "race", "gender",
                       "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


# Retrieve data
train_file_path = os.path.join(FLAGS.data_dir, "adult.data")
if os.path.isfile(train_file_path):
  train_file = open(train_file_path)
else:
  train_file = tempfile.NamedTemporaryFile()
  urllib.urlretrieve(TRAIN_DATA_URL, train_file.name)

test_file_path = os.path.join(FLAGS.data_dir, "adult.test")
if os.path.isfile(test_file_path):
  test_file = open(test_file_path)
else:
  test_file = tempfile.NamedTemporaryFile()
  urllib.urlretrieve(TEST_DATA_URL, test_file.name)

# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True,
                      skiprows=1)

# Remove the NaN values in the last rows of the tables
df_train = df_train[:-1]
df_test = df_test[:-1]

df_train[LABEL_COLUMN] = (
    df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (
    df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)


# TODO(cais): Turn into minibatch feeder
def input_fn(df):
  """Input data function.

  Creates a dictionary mapping from each continuous feature column name
  (k) to the values of that column stored in a constant Tensor.

  Args:
    df: data feed

  Returns:
    feature columns and labels
  """

  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def _create_experiment_fn(output_dir):
  """Experiment creation function."""
  # TODO(cais): Sanity check on flags?

  config = run_config.RunConfig(master=FLAGS.master_grpc_url,
                                num_ps_replicas=FLAGS.num_parameter_servers,
                                task=FLAGS.worker_index)

  estimator = tf.contrib.learn.DNNLinearCombinedClassifier(
      model_dir=FLAGS.model_dir,  # TODO(cais): How about the OSS distributed?
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=[5],
      config=config)  # cais: Use very small layer sizes to speed up testing.

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=lambda: input_fn(df_train),
      eval_input_fn=lambda: input_fn(df_test),
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps
  )


def main(unused_argv):
  print("Worker index: %d" % FLAGS.worker_index)  # DEBUG(cais)
  learn_runner.run(experiment_fn=_create_experiment_fn)


if __name__ == "__main__":
  tf.app.run()
