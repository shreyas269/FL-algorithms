### !pip install --quiet --upgrade tensorflow-federated

# !pip install tensorflow-federated
# !pip install --quiet --upgrade nest-asyncio

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

def _simple_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return scaffold_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss)

def _create_one_client_state():
  return scaffold_tf.ClientState(client_index=-1, iters_count=0, client_controls=get_model_weights(_simple_model_fn()))

federated_algorithm = build_federated_averaging_process(_simple_model_fn, _create_one_client_state)

federated_data_type = federated_algorithm.next.type_signature.parameter[1]

server_state = federated_algorithm.initialize()

def deterministic_batch():
    return collections.OrderedDict(x=np.ones([1, 28, 28, 1], dtype=np.float32), y=np.ones([1], dtype=np.int32))

batch = tff.tf_computation(deterministic_batch)()
federated_data = [[batch], [batch, batch]]
client_states = [_create_one_client_state(), _create_one_client_state()]

loss_list = []
for i in range(5):
    server_state, loss, client_states = federated_algorithm.next(server_state, federated_data, client_states)
    loss_list.append(loss)
    print("round", i, " loss=", loss)
