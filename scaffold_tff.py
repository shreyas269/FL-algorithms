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


def _initialize_optimizer_vars(model, optimizer):
  """Creates optimizer variables to assign the optimizer's state."""
  # Create zero gradients to force an update that doesn't modify.
  # Force eagerly constructing the optimizer variables. Normally Keras lazily
  # creates the variables on first usage of the optimizer. Optimizers such as
  # Adam, Adagrad, or using momentum need to create a new set of variables shape
  # like the model weights.
  model_weights = get_model_weights(model)
  zero_gradient = [tf.zeros_like(t) for t in model_weights.trainable]
  optimizer.apply_gradients(zip(zero_gradient, model_weights.trainable))
  assert optimizer.variables()


def build_federated_averaging_process(
    model_fn,
    client_state_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    client_state_fn: A no-arg function that returns a
      `scaffold_tf.ClientState`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for server update.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  whimsy_model = model_fn()

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model_weights=get_model_weights(model),
        optimizer_state=server_optimizer.variables(),
        round_num=0,
        total_iters_count=0,
        server_controls=get_model_weights(model))

  server_state_type = server_init_tf.type_signature.result

  model_weights_type = server_state_type.model_weights

  client_state_type = tff.framework.type_from_tensors(client_state_fn())

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      client_state_type.iters_count, client_state_type.client_controls)  # pytype: disable=attribute-error  # gen-stub-imports
  def server_update_fn(server_state, model_delta, total_iters_count, round_client_controls):
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta,
                         total_iters_count, round_client_controls)

  @tff.tf_computation(server_state_type)
  def server_message_fn(server_state):
    return build_server_broadcast_message(server_state)

  server_message_type = server_message_fn.type_signature.result
  tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)

  @tff.tf_computation(tf_dataset_type, client_state_type, server_message_type)
  def client_update_fn(tf_dataset, client_state, server_message):
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    return client_update(model, tf_dataset, client_state, server_message,
                         client_optimizer)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)

  federated_client_state_type = tff.type_at_clients(client_state_type)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type,
                             federated_client_state_type)
  def run_one_round(server_state, federated_dataset, client_states):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `scaffold_tf.ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.
      client_states: A federated `scaffold_tf.ClientState`.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_states, server_message_at_client))

    weight_denom = client_outputs.client_weight
    round_model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=weight_denom)

    round_client_controls = tff.federated_mean(
        client_outputs.client_state.client_controls, weight=weight_denom)

    total_iters_count = tff.federated_sum(
        client_outputs.client_state.iters_count)

    server_state = tff.federated_map(
        server_update_fn, (server_state, round_model_delta, total_iters_count, round_client_controls))
    round_loss_metric = tff.federated_mean(
        client_outputs.model_output, weight=weight_denom)

    return server_state, round_loss_metric, client_outputs.client_state

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round)
