"""SCAFFOLD RUN
"""

import nest_asyncio
nest_asyncio.apply()

import collections
import functools
from typing import Callable, List, OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import scaffold_tf
from scaffold_tf import build_server_broadcast_message
from scaffold_tf import client_update
from scaffold_tf import get_model_weights
from scaffold_tf import server_update
from scaffold_tf import ServerState

from scaffold_tff import build_federated_averaging_process


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model

def _create_test_cnn_model_zero_weights():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer='zeros')

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model

def _simple_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return scaffold_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss)
  
def _simple_model_fn_zero_weights():
  keras_model = _create_test_cnn_model_zero_weights()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return scaffold_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


def _create_one_client_state():
  return scaffold_tf.ClientState(client_index=-1, iters_count=0, client_controls=get_model_weights(_simple_model_fn_zero_weights()))

federated_algorithm = build_federated_averaging_process(_simple_model_fn, _create_one_client_state)

federated_data_type = federated_algorithm.next.type_signature.parameter[1]

server_state = federated_algorithm.initialize()

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

NUM_CLIENTS = 100
BATCH_SIZE = 10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch of EMNIST data and return a (features, label) tuple."""
    return (tf.reshape(element['pixels'], [-1, 28, 28, 1]), 
            element['label'])

  return dataset.batch(BATCH_SIZE).map(batch_format_fn)

client_ids = sorted(emnist_train.client_ids)[:NUM_CLIENTS]
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x))
  for x in client_ids
]

client_states = [_create_one_client_state()] * NUM_CLIENTS

scaffold_loss_list = []
for i in range(50):
    server_state, loss, client_states = federated_algorithm.next(server_state, federated_train_data, client_states)
    scaffold_loss_list.append(loss)
    print("round", i+1, " loss=", loss)
