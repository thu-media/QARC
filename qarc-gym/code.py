# -*- coding: utf-8 -*-
"""code.py: sourcecode for reward redistribution tutorial

Author -- Michael Widrich
Contact -- widrich@bioinf.jku.at

"""
import os
import datetime as dt
from collections import OrderedDict
import numpy as np

import tensorflow as tf
from TeLL.layers import DenseLayer, LSTMLayer, RNNInputLayer, ConcatLayer, ReshapeLayer
from TeLL.utility.misc import make_sure_path_exists
from TeLL.utility.misc_tensorflow import tensor_shape_with_flexible_dim, TriangularValueEncoding
from TeLL.utility.plotting import save_subplots_line_plots
from TeLL.regularization import regularize

rnd_seed = 123
rnd_gen = np.random.RandomState(seed=rnd_seed)
tf.set_random_seed(rnd_seed)

#
# Set up an example environment
#
max_timestep = 50
n_mb = 1
n_features = 13
n_actions = 2
ending_frames = 10


def generate_sample():
    """Create sample episodes from our example environment"""
    # Create random actions
    actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(max_timestep,)), dtype=np.float32)
    actions_onehot = np.zeros((max_timestep, 2), dtype=np.float32)
    actions_onehot[actions == 0, 0] = 1
    actions_onehot[:, 1] = 1 - actions_onehot[:, 0]
    actions += actions - 1
    # Create states to actions, make sure agent stays in range [-6, 6]
    states = np.zeros_like(actions)
    for i, a in enumerate(actions):
        if i == 0:
            states[i] = a
        else:
            states[i] = np.clip(states[i - 1] + a, a_min=-int(n_features / 2), a_max=int(n_features / 2))
    
    # Check when agent collected a coin (=is at position 2)
    coin_collect = np.asarray(states == 2, dtype=np.float32)
    # Move all reward to position 50 to make it a delayed reward example
    coin_collect[-1] = np.sum(coin_collect)
    coin_collect[:-1] = 0
    rewards = coin_collect
    # Padd end of game sequences with zero-states
    states = np.asarray(states, np.int) + int(n_features / 2)
    states_onehot = np.zeros((len(rewards) + ending_frames, n_features), dtype=np.float32)
    states_onehot[np.arange(len(rewards)), states] = 1
    actions_onehot = np.concatenate((actions_onehot, np.zeros_like(actions_onehot[:ending_frames])))
    rewards = np.concatenate((rewards, np.zeros_like(rewards[:ending_frames])))
    # Return states, actions, and rewards
    return dict(states=states_onehot[None, :], actions=actions_onehot[None, :], rewards=rewards[None, :, None])


#
# Set up reward redistribution model
#
state_shape_per_ts = (n_mb, 1, n_features)
action_shape_per_ts = (n_mb, 1, n_actions)
# This will encode the in-game time in multiple nodes
timestep_encoder = TriangularValueEncoding(max_value=max_timestep, triangle_span=int(max_timestep / 10))

states_placeholder = tf.placeholder(shape=(n_mb, None, n_features), dtype=tf.float32)
actions_placeholder = tf.placeholder(shape=(n_mb, None, n_actions), dtype=tf.float32)
rewards_placeholder = tf.placeholder(shape=(n_mb, None, 1), dtype=tf.float32)
n_timesteps = tf.shape(rewards_placeholder)[1] - 1

with tf.variable_scope('reward_redistribution_model', reuse=tf.AUTO_REUSE):
    state_input_layer = RNNInputLayer(tf.zeros(state_shape_per_ts, dtype=tf.float32))
    action_input_layer = RNNInputLayer(tf.zeros(action_shape_per_ts, dtype=tf.float32))
    time_input_layer = RNNInputLayer(timestep_encoder.encode_value(tf.constant(0, dtype=tf.int32)))
    time_input = ReshapeLayer(incoming=time_input_layer, shape=(n_mb, 1, timestep_encoder.n_nodes_python))
    reward_redistibution_input = ConcatLayer(incomings=[state_input_layer, action_input_layer, time_input],
                                             name='RewardRedistributionInput')
    n_lstm_cells = 8
    truncated_normal_init = lambda mean, stddev: \
        lambda *args, **kwargs: tf.truncated_normal(mean=mean, stddev=stddev, *args, **kwargs)
    w_init = truncated_normal_init(mean=0, stddev=1)
    og_bias = truncated_normal_init(mean=0, stddev=1)
    ig_bias = truncated_normal_init(mean=-1, stddev=1)
    ci_bias = truncated_normal_init(mean=0, stddev=1)
    fg_bias = truncated_normal_init(mean=-5, stddev=1)
    lstm_layer = LSTMLayer(incoming=reward_redistibution_input, n_units=n_lstm_cells, name='LSTMRewardRedistribution',
                           W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                           b_ci=ci_bias([n_lstm_cells]), b_ig=ig_bias([n_lstm_cells]),
                           b_og=og_bias([n_lstm_cells]), b_fg=fg_bias([n_lstm_cells]),
                           a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.identity,
                           c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, store_states=True)
    
    n_output_units = 4
    output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, a=tf.identity, W=w_init,
                              b=tf.zeros([n_output_units], dtype=tf.float32), name="OutputLayer")

lstm_input_shape = reward_redistibution_input.get_output_shape()


#
# Ending condition
#
def cond(time, *args):
    """Break if game is over by looking at n_timesteps"""
    return ~tf.greater(time, n_timesteps)


#
# Loop body
#
# Create initial tensors
init_tensors = OrderedDict([
    ('time', tf.constant(0, dtype=tf.int32)),
    ('lstm_inputs', tf.zeros(lstm_input_shape)),
    ('lstm_internals', tf.expand_dims(tf.stack([lstm_layer.c[-1], lstm_layer.c[-1],
                                                lstm_layer.c[-1], lstm_layer.c[-1],
                                                lstm_layer.c[-1]], axis=-1), axis=1)),
    ('lstm_h', tf.expand_dims(lstm_layer.h[-1], axis=1)),
    ('predictions', tf.zeros([s if s >= 0 else 1 for s in output_layer.get_output_shape()]))
])

# Get initial tensor shapes in tf format
init_shapes = OrderedDict([
    ('time', init_tensors['time'].get_shape()),
    ('lstm_inputs', tensor_shape_with_flexible_dim(init_tensors['lstm_inputs'], dim=1)),
    ('lstm_internals', tensor_shape_with_flexible_dim(init_tensors['lstm_internals'], dim=1)),
    ('lstm_h', tensor_shape_with_flexible_dim(init_tensors['lstm_h'], dim=1)),
    ('predictions', tensor_shape_with_flexible_dim(init_tensors['predictions'], dim=1)),
])


def body(time, lstm_inputs, lstm_internals, lstm_h, predictions, *args):
    """Loop over states and additional inputs, compute network outputs and store hidden states and activations for
    debugging/plotting"""
    
    # Set states as network input
    state_input_layer.update(tf.expand_dims(states_placeholder[:, time], axis=1))
    
    # Set actions as network input
    action_input_layer.update(tf.cast(tf.expand_dims(actions_placeholder[:, time], axis=1), dtype=tf.float32))
    
    # Set time as network input
    time_input_layer.update(timestep_encoder.encode_value(time))
    
    # Update LSTM cell-state and output with states from last timestep
    lstm_layer.c[-1], lstm_layer.h[-1] = lstm_internals[:, -1, :, -1], lstm_h[:, -1, :]
    
    # Calculate reward redistribution network output and append it to last timestep
    predictions = tf.concat([predictions, output_layer.get_output()], axis=1)
    
    # Store LSTM states for all timesteps for visualization
    lstm_internals = tf.concat([lstm_internals,
                                tf.expand_dims(tf.stack([lstm_layer.ig[-1], lstm_layer.og[-1],
                                                         lstm_layer.ci[-1], lstm_layer.fg[-1],
                                                         lstm_layer.c[-1]], axis=-1), axis=1)],
                               axis=1)
    lstm_h = tf.concat([lstm_h, tf.expand_dims(lstm_layer.h[-1], axis=1)], axis=1)
    
    # Increment time
    time += tf.constant(1, dtype=tf.int32)
    
    lstm_inputs = tf.concat([lstm_inputs, reward_redistibution_input.out], axis=1)
    
    return [time, lstm_inputs, lstm_internals, lstm_h, predictions]


wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=tuple(init_tensors.values()),
                       shape_invariants=tuple(init_shapes.values()),
                       parallel_iterations=50, back_prop=True, swap_memory=True)

# Re-Associate returned tensors with keys
rr_returns = OrderedDict(zip(init_tensors.keys(), wl_ret))

# Remove initialization timestep
rr_returns['lstm_internals'] = rr_returns['lstm_internals'][:, 1:]
rr_returns['lstm_h'] = rr_returns['lstm_h'][:, 1:]
rr_returns['predictions'] = rr_returns['predictions'][:, 1:]
lstm_inputs = rr_returns['lstm_inputs'][:, 1:]

#
# Define updates
#
aux_target_1_filter = tf.ones((10, 1, 1), dtype=tf.float32)
aux_target_1 = tf.concat([rewards_placeholder, tf.zeros_like(rewards_placeholder[:, :9])], axis=1)
aux_target_1 = tf.nn.conv1d(aux_target_1, filters=aux_target_1_filter, padding='VALID', stride=1)
aux_target_2 = tf.reduce_sum(rewards_placeholder, axis=1) - tf.cumsum(rewards_placeholder, axis=1)
aux_target_3 = tf.cumsum(rewards_placeholder, axis=1)
targets = tf.concat([rewards_placeholder, aux_target_1, aux_target_2, aux_target_3], axis=-1)

return_prediction = rr_returns['predictions'][0, -1, 0]
true_return = tf.reduce_sum(targets[0, :, 0])
reward_prediction_error = tf.square(true_return - return_prediction)
auxiliary_losses = tf.reduce_mean(tf.square(targets[0, :, 1:] - rr_returns['predictions'][0, :, 1:]),
                                  axis=1)
# Add regularization penalty
rr_reg_penalty = regularize(layers=[lstm_layer, output_layer], l1=1e-6, regularize_weights=True, regularize_biases=True)
total_loss = (reward_prediction_error + tf.reduce_mean(auxiliary_losses)) / 2 + rr_reg_penalty

trainables = tf.trainable_variables()
grads = tf.gradients(total_loss, trainables)
grads, _ = tf.clip_by_global_norm(grads, 0.5)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
rr_update = optimizer.apply_gradients(zip(grads, trainables))

#
# Set up Integrated Gradients for contribution analysis
#
n_intgrd_steps = 500
# Create multiplier for interpolating between full and zeroed input sequence
intgrd_w = np.linspace(0, 1, num=n_intgrd_steps, dtype=np.float32)
intgrd_inputs = tf.concat([lstm_inputs * w for w in intgrd_w], axis=0)
intgrd_input_shape_per_ts = intgrd_inputs.shape.as_list()
intgrd_input_shape_per_ts[1] = 1

with tf.variable_scope('reward_redistribution_model', reuse=tf.AUTO_REUSE):
    reward_redistibution_input = RNNInputLayer(tf.zeros(intgrd_input_shape_per_ts))
    
    lstm_layer = LSTMLayer(incoming=reward_redistibution_input, n_units=n_lstm_cells, name='LSTMRewardRedistribution',
                           W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                           b_ci=ci_bias([n_lstm_cells]), b_ig=ig_bias([n_lstm_cells]),
                           b_og=og_bias([n_lstm_cells]), b_fg=fg_bias([n_lstm_cells]),
                           a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.identity,
                           c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, store_states=True)
    
    n_output_units = 4
    output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, a=tf.identity, W=w_init,
                              b=tf.zeros([n_output_units], dtype=tf.float32), name="OutputLayer")


#
# Ending condition
#
def cond(time, *args):
    """Break if game is over by looking at n_timesteps"""
    return ~tf.greater(time, n_timesteps)


#
# Loop body
#
# Create initial tensors
init_tensors = OrderedDict([
    ('time', tf.constant(0, dtype=tf.int32)),
    ('lstm_c', tf.zeros((n_intgrd_steps, n_lstm_cells), dtype=tf.float32)),
    ('lstm_h', tf.zeros((n_intgrd_steps, n_lstm_cells), dtype=tf.float32)),
    ('predictions', tf.zeros(output_layer.get_output_shape()))
])

# Get initial tensor shapes in tf format
init_shapes = OrderedDict([
    ('time', init_tensors['time'].get_shape()),
    ('lstm_c', init_tensors['lstm_c'].get_shape()),
    ('lstm_h', init_tensors['lstm_h'].get_shape()),
    ('predictions', init_tensors['predictions'].get_shape()),
])


def body(time, lstm_c, lstm_h, predictions, *args):
    """Loop over states and additional inputs, compute network outputs and store hidden states and activations for
    debugging/plotting"""
    
    # Set model input
    reward_redistibution_input.update(tf.expand_dims(intgrd_inputs[:, time], axis=1))
    
    # Update LSTM cell-state and output with states from last timestep
    lstm_layer.c[-1], lstm_layer.h[-1] = lstm_c, lstm_h
    
    # Calculate reward redistribution network output and append it to last timestep
    predictions = output_layer.get_output()
    
    # Store LSTM states for all timesteps for visualization
    lstm_c = lstm_layer.c[-1]
    lstm_h = lstm_layer.h[-1]
    
    # Increment time
    time += tf.constant(1, dtype=tf.int32)
    
    return [time, lstm_c, lstm_h, predictions]


wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=tuple(init_tensors.values()),
                       shape_invariants=tuple(init_shapes.values()),
                       parallel_iterations=50, back_prop=True, swap_memory=True)

# Re-Associate returned tensors with keys
ig_returns = OrderedDict(zip(init_tensors.keys(), wl_ret))

# For reward redistribution, use only the prediction for main task and aux. task for accumulated reward prediction at
# the last timestep
intgrd_pred = ig_returns['predictions'][..., 0] + ig_returns['predictions'][..., -1]

# Get gradients, set NaNs to 0
grads = tf.gradients(intgrd_pred, intgrd_inputs)[0]
grads = tf.where(tf.is_nan(grads), tf.zeros_like(grads), grads)
# Calc gradients, sum over batch dimension
intgrd_grads = tf.reduce_sum(grads, axis=0)
# Scale by original input
intgrd_grads *= lstm_inputs[0]
# Sum over features
intgrd_grads = tf.reduce_sum(intgrd_grads, axis=-1) / n_intgrd_steps
# Set last timesteps to 0 to avoid numerical IG problems
intgrd_grads = tf.concat([intgrd_grads[:-10], tf.zeros_like(intgrd_grads[:10])], axis=0)

# Calculate quality of IG
intgrd_zero_prediction = intgrd_pred[0]
intgrd_full_prediction = intgrd_pred[-1]
intgrd_prediction_diff = intgrd_full_prediction - intgrd_zero_prediction
intgrd_sum = tf.reduce_sum(intgrd_grads)
# Compute percentage of integrated gradients reconstruction quality
#  (in case 0-sequence and full-sequence are equal, take a heuristic to decide on intgrdperc)
intgrd_reconstruction_perc = tf.cond(tf.less(tf.abs(intgrd_prediction_diff), tf.constant(1e-5, dtype=tf.float32))[0],
                                     lambda: 100. + intgrd_sum,
                                     lambda: (100. / intgrd_prediction_diff) * intgrd_sum)

# Correct integrated gradients signal for prediction error and int. grds error:
avg_reward_placeholder = tf.placeholder(shape=(), dtype=tf.float32)
error = return_prediction - intgrd_sum
intgrd_grads += error / true_return
epsilon_sqr = tf.sqrt(tf.clip_by_value(np.abs(avg_reward_placeholder), clip_value_min=1e-5,
                                       clip_value_max=np.abs(avg_reward_placeholder)))
intgrd_grads *= tf.clip_by_value(true_return /
                                 (return_prediction + (tf.sign(return_prediction) * (tf.sqrt(epsilon_sqr) / 5))),
                                 clip_value_min=1e-5, clip_value_max=1.5)

#
# Train reward redistribution model and get integrated gradients signal
#
# Tensorflow configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
)
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.global_variables_initializer().run(session=tf_sess)

outputpath = os.path.join('results', dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
outputpath_training = os.path.join(outputpath, 'training')
make_sure_path_exists(outputpath)
make_sure_path_exists(outputpath_training)
avg_return = 0.
n_max_updates = 1e5
episode = 0
n_plotted = 0
avg_loss = 0
while episode < n_max_updates:
    sample = generate_sample()
    
    feed_dict = {states_placeholder: sample['states'],
                 actions_placeholder: sample['actions'],
                 rewards_placeholder: sample['rewards'],
                 avg_reward_placeholder: avg_return}
    loss, true_ret, pred_ret, _ = tf_sess.run([total_loss, true_return, return_prediction, rr_update],
                                              feed_dict=feed_dict)
    avg_return = avg_return * 0.99 + true_ret * 0.01
    avg_loss = avg_loss * 0.99 + loss * 0.01
    
    if episode % 100 == 0:
        print(
                "episode {}: loss {}; avg_loss {}; avg_ret {}; ret {}; pred {};".format(episode, loss, avg_loss,
                                                                                        avg_return,
                                                                                        true_ret, pred_ret))
        
        if episode % 1000 == 0:
            lstm_internal_values, lstm_h_values, target_values, prediction_values = \
                tf_sess.run([rr_returns['lstm_internals'][0], rr_returns['lstm_h'][0], targets[0],
                             rr_returns['predictions'][0]], feed_dict=feed_dict)
            save_subplots_line_plots(
                    images=([np.argmax(sample['states'][0], axis=-1) - int(n_features / 2), prediction_values[..., 0],
                             target_values[..., 1], prediction_values[..., 1],
                             target_values[..., 2], prediction_values[..., 2],
                             target_values[..., 3], prediction_values[..., 3]]),
                    subfigtitles=(['states', 'return pred. (only last timestep)',
                                   'true aux1', 'pred aux1', 'true aux2', 'pred aux2', 'true aux 3', 'pred aux3']),
                    automatic_positioning=True, tight_layout=True,
                    filename=os.path.join(outputpath_training, "predictions_ep{}.png".format(episode)))
            
            save_subplots_line_plots(
                    images=[lstm_internal_values[:, :, 0], lstm_internal_values[:, :, 1],
                            lstm_internal_values[:, :, 2], lstm_internal_values[:, :, 3],
                            lstm_internal_values[:, :, 4], lstm_h_values[:, :]],
                    subfigtitles=['rr_lstm_ig', 'rr_lstm_og', 'rr_lstm_ci', 'rr_lstm_fg', 'rr_lstm_c', 'rr_lstm_h'],
                    automatic_positioning=True, tight_layout=True,
                    filename=os.path.join(outputpath_training, "lstm_internals_ep{}.png".format(episode)))
    
    if avg_loss < 1 and true_ret != 0 and episode > 5000:
        print("\tperforming IG, plotting curves...")
        intgrd_grads_value, intgrd_reconstruction_perc_value = tf_sess.run([intgrd_grads, intgrd_reconstruction_perc],
                                                                           feed_dict=feed_dict)
        print("\tIG quality: {}%".format(intgrd_reconstruction_perc_value))
        if 80 < intgrd_reconstruction_perc_value < 120:
            print([(s, a, i) for s, a, i in zip(np.argmax(sample['states'][0, :, :], axis=-1),
                                                np.argmax(sample['states'][0], axis=-1) - int(n_features / 2),
                                                intgrd_grads_value)])
            n_plotted += 1
            
            target_values, prediction_values = tf_sess.run([targets[0], rr_returns['predictions'][0]],
                                                           feed_dict=feed_dict)
            save_subplots_line_plots(
                    images=[np.argmax(sample['states'][0], axis=-1) - int(n_features / 2), intgrd_grads_value,
                            np.argmax(sample['actions'][0], axis=-1), sample['rewards'][0]],
                    subfigtitles=['states',
                                  'redistributed reward (IG reconstructed {}%)'.format(
                                      intgrd_reconstruction_perc_value),
                                  'taken actions', 'rewards'],
                    automatic_positioning=True, tight_layout=True,
                    filename=os.path.join(outputpath, "reward_redistribution_ep{}.png".format(episode)))
            
            save_subplots_line_plots(
                    images=([target_values[..., 0], prediction_values[..., 0],
                             target_values[..., 1], prediction_values[..., 1],
                             target_values[..., 2], prediction_values[..., 2],
                             target_values[..., 3], prediction_values[..., 3]]),
                    subfigtitles=(['return (only last timestep)', 'return pred. (only last timestep)',
                                   'true aux1', 'pred aux1', 'true aux2', 'pred aux2', 'true aux 3', 'pred aux3']),
                    automatic_positioning=True, tight_layout=True,
                    filename=os.path.join(outputpath, "predictions_ep{}.png".format(episode)))
            
            if n_plotted > 50:
                print("Plotted enough examples")
                break
    episode += 1

tf_sess.close()
print("Done")

