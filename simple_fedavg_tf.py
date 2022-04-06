import attr
import tensorflow as tf
import tensorflow_federated as tff


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Attributes:
    weights_delta: A dictionary of updates to the model's trainable variables.
    client_weight: Weight to be used in a weighted mean when aggregating
      `weights_delta`.
    model_output: A structure matching
      `tff.learning.Model.report_local_unfinalized_metrics`, reflecting the
      results of training on the input dataset.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.

  Attributes:
    model:  A `tff.learning.ModelWeights` structure, containing Tensors or
      Variables.
    optimizer_state: Variables of optimizer.
    round_num: The current round in the training process.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
  """Structure for tensors broadcasted by server during federated optimization.

  Attributes:
    model_weights: A `tff.learning.ModelWeights` structure, containing Tensors
      or Variables.
    round_num: Round index to broadcast. We use `round_num` as an example to
      show how to broadcast auxiliary information that can be helpful on
      clients. It is not explicitly used, but can be applied to enable learning
      rate scheduling.
  """
  model_weights = attr.ib()
  round_num = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  # Initialize the model with the current state.
  model_weights = tff.learning.ModelWeights.from_model(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model)
  tf.nest.map_structure(lambda v, t: v.assign(t), server_optimizer.variables(),
                        server_state.optimizer_state)

  # Apply the update to the model.
  neg_weights_delta = [-1.0 * x for x in weights_delta]
  server_optimizer.apply_gradients(
      zip(neg_weights_delta, model_weights.trainable), name='server_update')

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
  """Builds `BroadcastMessage` for broadcasting.

  This method can be used to post-process `ServerState` before broadcasting.
  For example, perform model compression on `ServerState` to obtain a compressed
  state that is sent in a `BroadcastMessage`.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model, round_num=server_state.round_num)


@tf.function
def client_update(model, dataset, server_message, client_optimizer):
  """Performans client local training of `model` on `dataset`.

  Args:
    model: A `tff.learning.Model` to train locally on the client.
    dataset: A 'tf.data.Dataset' representing the clients local dataset.
    server_message: A `BroadcastMessage` from serve containing the initial
      model weights to train.
    client_optimizer: A `tf.keras.optimizers.Optimizer` used to update the local
      model during training.

  Returns:
    A `ClientOutput` instance with a model update to aggregate on the server.
  """
  model_weights = tff.learning.ModelWeights.from_model(model)
  initial_weights = server_message.model_weights
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        initial_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  # Explicit use `iter` for dataset is a trick that makes TFF more robust in
  # GPU simulation and slightly more performant in the unconventional usage
  # of large number of small datasets.
  for batch in iter(dataset):
    with tf.GradientTape() as tape:
      outputs = model.forward_pass(batch)
    grads = tape.gradient(outputs.loss, model_weights.trainable)
    client_optimizer.apply_gradients(zip(grads, model_weights.trainable))
    batch_size = tf.shape(batch['y'])[0]
    num_examples += batch_size

  weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                        model_weights.trainable,
                                        initial_weights.trainable)
  client_weight = tf.cast(num_examples, tf.float32)
  model_outputs = model.report_local_unfinalized_metrics()
  return ClientOutput(weights_delta, client_weight, model_outputs)
