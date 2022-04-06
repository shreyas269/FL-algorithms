import collections
from typing import Any, Collection, Dict, Optional

import attr

import tensorflow as tf
import tensorflow_federated as tff

def _unpack_data_label(batch):
  if isinstance(batch, collections.abc.Mapping):
    return batch['x'], batch['y']
  elif isinstance(batch, (tuple, list)):
    if len(batch) < 2:
      raise ValueError('Expecting both data and label from a batch.')
    return batch[0], batch[1]
  else:
    raise ValueError('Unrecognized batch data.')

@attr.s(eq=False)
class OptimizerState(object):
  iterations = attr.ib()
  weights = attr.ib()


def _initialize_optimizer_vars(model, optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model):
  if hasattr(model, 'weights'):
    return model.weights
  else:
    return tff.learning.ModelWeights.from_model(model)


def _get_optimizer_state(optimizer):
  return OptimizerState(
      iterations=optimizer.iterations,
      # The first weight of an optimizer is reserved for the iterations count,
      # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/optimizer_v2.py  pylint: disable=line-too-long]
      weights=tuple(optimizer.weights[1:]))


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Attributes:
    model: A `tff.learning.ModelWeights` instance.
    optimizer_state: A namedtuple of the optimizer variables.
    round_num: The current training round, as a float.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.

class CreateServerUpdateFn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """

  def __init__(self):
    self.random_generator = tf.random.Generator.from_non_deterministic_state()

  def _noise_fn(self, noise_std: float, model_weight_specs: Collection[tf.TensorSpec]):
    """Returns random noise to be added for differential privacy."""

    def noise_tensor(spec):
      noise = self.random_generator.normal(spec.shape, stddev=noise_std)
      noise = tf.reshape(noise, spec.shape)
      return noise

    return tf.nest.map_structure(noise_tensor, model_weight_specs)

  @tf.function
  def __call__(self, model,
                    optimizer,
                    server_state,
                    weights_delta,
                    server_learning_rate=1.0):
    """Updates `server_state` based on `weights_delta`, increase the round number.

    Args:
      model: A `tff.learning.Model`.
      optimizer: A `tf.keras.optimizers.Optimizer`.
      server_state: A `ServerState`, the state to be updated.
      weights_delta: An update to the trainable variables of the model.
      server_learning_rate: Server learning rate scales the update from clients
        before applying to server. Defaults to 1.

    Returns:
      An updated `ServerState`.
    """
    model_weights = _get_weights(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          server_state.model)

    model_weight_specs = tf.nest.map_structure(
        lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights.trainable)

    # Compute new model weights.
    new_weights = tf.nest.map_structure(lambda a, b: a + server_learning_rate * b,
                                        model_weights.trainable, weights_delta)

    # Set the model weights to the new ones, overriding the update made by
    # the optimizer.
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights.trainable,
                          new_weights)

    # Create a new state based on the updated model.
    return tff.structure.update_struct(
        server_state,
        model=model_weights,
        round_num=server_state.round_num)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Attributes:
    weights_delta: A dictionary of updates to the model's trainable variables.
    client_weight: Weights to be used in a weighted mean when aggregating
      `weights_delta`.
    model_output: A structure matching `tff.learning.Model.report_local_outputs`
      reflecting the results of training on the input dataset.
    optimizer_output: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


class CreateClientUpdateFn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """

  def __init__(self):
    self.grad_sum = None

  @tf.function
  def __call__(self,
               model,
               dataset,
               initial_weights,
               initial_optimizer_state,
               optimizer,
               client_weight_fn=None):
    """Updates client model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.ModelWeights` from server.
      initial_optimizer_state: The variables to assign to the client optimizer.
      optimizer: A `tf.keras.optimizer.Optimizer` object, assumed to be
        identical to the optimizer used by the server.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.

    Returns:
      A 'ClientOutput`.
    """

    model_weights = _get_weights(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          initial_weights)

    # Compute gradient over full data at initial_weights.
    # This assumes that the loss is an average over all examples in a batch,
    # and that all batches have the same size (otherwise, last batch has a
    # slightly higher weight).
    num_batches = 0.0
    loss_sum = 0.0
    # Client optimizer variables must be initialized prior to invoking this
    optimizer_state = _get_optimizer_state(optimizer)

    num_examples = tf.constant(0, dtype=tf.int32)

    for batch in iter(dataset):
      # keep optimizer state fixed to initial values.
      tf.nest.map_structure(lambda v, t: v.assign(t), optimizer_state,
                            initial_optimizer_state)
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights.trainable)
      grads_and_vars = zip(grads, model_weights.trainable)
      optimizer.apply_gradients(grads_and_vars)
      if hasattr(output, 'num_examples'):
        batch_size = tf.cast(output.num_examples, dtype=tf.int32)
      else:
        batch_x, _ = _unpack_data_label(batch)
        batch_size = tf.shape(batch_x)[0]

      num_examples+=batch_size
      loss_sum += output.loss * tf.cast(batch_size, tf.float32)

    aggregated_outputs = loss_sum
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)

    if client_weight_fn is None:
      client_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    optimizer_output = collections.OrderedDict([('num_examples', num_examples)])

    return ClientOutput(
        weights_delta=weights_delta,
        client_weight=client_weight,
        model_output=loss_sum / client_weight,
        optimizer_output=optimizer_output)


def build_server_init_fn(model_fn, optimizer_fn, base_lr, server_momentum):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model`, `ServerState.optimizer_state`, and
  `ServerState.optimizer_state` are initialized via their constructor
  functions. The attribute `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    base_lr: Learning rate for server optimizer
    server_momentum: Momentum for server optimizer

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    optimizer = optimizer_fn(learning_rate=base_lr,momentum=server_momentum)

    model = model_fn()
    _initialize_optimizer_vars(model, optimizer)
    return ServerState(
        model=_get_weights(model),
        optimizer_state=_get_optimizer_state(optimizer),
        round_num=0)


  return server_init_tf


def build_federated_averaging_process(model_fn,
                           optimizer_fn=tf.keras.optimizers.SGD,
                           base_lr=0.1,
                           server_lr=1.0,
                           server_momentum=0.0,
                           client_weight_fn=None):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer_fn: A function that accepts a `learning_rate` argument and returns
      a `tf.keras.optimizers.Optimizer` instance. Must return an optimizer with
      `iterations` and `weights` attributes. This is the base optimizer whose
      updates are split between the client and server in the Mime/Mimelite
      algorithms.
    base_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate for the base optimizer.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate for applying weight
      updates to server model.
    server_momentum: A scalar momentum parameter for the server optimizer.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of the client models. If not provided, the
      default is the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  base_lr_schedule = base_lr
  if not callable(base_lr_schedule):
    base_lr_schedule = lambda round_num: base_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  dummy_model = model_fn()


  server_init_tf = build_server_init_fn(model_fn, optimizer_fn, server_lr, server_momentum)
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  optimizer_state_type = server_state_type.optimizer_state
  round_num_type = server_state_type.round_num

  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)

  federated_dataset_type = tff.type_at_clients(tf_dataset_type)

  @tff.tf_computation(model_input_type, model_weights_type,
                      optimizer_state_type, round_num_type)
  def client_update_fn(tf_dataset, initial_model_weights,
                       initial_optimizer_state, round_num):
    """Performs a client update."""
    model = model_fn()
    base_lr = base_lr_schedule(round_num)
    optimizer = optimizer_fn(learning_rate=base_lr, momentum=server_momentum)
    # We initialize the client optimizer variables to avoid creating them
    # within the scope of the tf.function client_update.
    _initialize_optimizer_vars(model, optimizer)

    client_update = CreateClientUpdateFn()
    return client_update(model, tf_dataset, initial_model_weights,
                         initial_optimizer_state, optimizer, client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    base_lr = base_lr_schedule(server_state.round_num)
    optimizer = optimizer_fn(learning_rate=base_lr, momentum=server_momentum)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, optimizer)
    server_update = CreateServerUpdateFn()
    return server_update(model, optimizer, server_state, model_delta,
                         server_lr)


  @tff.federated_computation(
      tff.type_at_server(server_state_type),
      tff.type_at_clients(tf_dataset_type))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)
    optimizer_state = tff.federated_broadcast(server_state.optimizer_state)
    client_round_num = tff.federated_broadcast(server_state.round_num)
    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_model, optimizer_state, client_round_num))

    weight_denom = client_outputs.client_weight

    model_delta = tff.federated_mean(
        client_outputs.weights_delta)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, model_delta))

    round_loss_metric = tff.federated_mean(
        client_outputs.model_output, weight=weight_denom)

    return server_state, round_loss_metric

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round)
