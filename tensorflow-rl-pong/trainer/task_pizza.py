# Copyright 2018 Google LLC
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


import argparse
import os
import tensorflow as tf
import numpy as np
from builtins import input

from trainer.helpers import discount_rewards, prepro
import trainer.gym_pizza as gym
from agents.tools.wrappers import AutoReset
from collections import deque


ROWS = 3
COLS = 5
OBSERVATION_DIM = ROWS * COLS
LOGITS_SIZE = 2 * ROWS + 2 * COLS

MEMORY_CAPACITY = 100000
ROLLOUT_SIZE = 10000


# MEMORY stores tuples:
# (observation, label, reward)
MEMORY = deque([], maxlen=MEMORY_CAPACITY)
def gen():
    for m in list(MEMORY):
        yield m


def build_graph(observations):
    """Calculates logits from the input observations tensor.
    This function will be called twice: rollout and train.
    The weights will be shared.
    """
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        observations = tf.layers.flatten(observations)
        hidden = tf.layers.dense(observations, args.hidden_dim, use_bias=False, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, LOGITS_SIZE, use_bias=False)

    return logits


def main(args):
    args_dict = vars(args)
    print('args: {}'.format(args_dict))

    with tf.Graph().as_default() as g:
        # rollout subgraph
        with tf.name_scope('rollout'):
            observations = tf.placeholder(shape=(None, ROWS, COLS), dtype=tf.float32)
            
            logits = build_graph(observations)

            logits_for_sampling = tf.reshape(logits, shape=(1, LOGITS_SIZE))

            logits_for_sampling_left = logits_for_sampling[:, :COLS]
            logits_for_sampling_right = logits_for_sampling[:, COLS:COLS+COLS]
            logits_for_sampling_top = logits_for_sampling[:, COLS+COLS:COLS+COLS+ROWS]
            logits_for_sampling_bottom = logits_for_sampling[:, COLS+COLS+ROWS:]

            # Sample the action to be played during rollout.
            sample_action_left = tf.squeeze(tf.multinomial(logits=logits_for_sampling_left, num_samples=1))
            sample_action_right = tf.squeeze(tf.multinomial(logits=logits_for_sampling_right, num_samples=1))
            sample_action_top = tf.squeeze(tf.multinomial(logits=logits_for_sampling_top, num_samples=1))
            sample_action_bottom = tf.squeeze(tf.multinomial(logits=logits_for_sampling_bottom, num_samples=1))

            sample_action = tf.stack([sample_action_left,
                                      sample_action_right,
                                      sample_action_top,
                                      sample_action_bottom], axis=0)
        
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=args.learning_rate,
            decay=args.decay
        )

        # dataset subgraph for experience replay
        with tf.name_scope('dataset'):
            # the dataset reads from MEMORY
            ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
            ds = ds.shuffle(MEMORY_CAPACITY).repeat().batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()

        # training subgraph
        with tf.name_scope('train'):
            # the train_op includes getting a batch of data from the dataset, so we do not need to use a feed_dict when running the train_op.
            next_batch = iterator.get_next()
            train_observations, labels, processed_rewards = next_batch

            # This reuses the same weights in the rollout phase.
            train_observations.set_shape((args.batch_size, OBSERVATION_DIM))
            train_logits = build_graph(train_observations)

            train_logits_left = train_logits[:, :COLS]
            train_logits_right = train_logits[:, COLS:COLS+COLS]
            train_logits_top = train_logits[:, COLS+COLS:COLS+COLS+ROWS]
            train_logits_bottom = train_logits[:, COLS+COLS+ROWS:]

            labels_left = labels[:, :COLS]
            labels_right = labels[:, COLS:COLS + COLS]
            labels_top = labels[:, COLS + COLS:COLS + COLS + ROWS]
            labels_bottom = labels[:, COLS + COLS + ROWS:]

            cross_entropies_left = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits_left,
                                                                                  labels=labels_left)
            cross_entropies_right = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits_right,
                                                                                   labels=labels_right)
            cross_entropies_top = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits_top,
                                                                                 labels=labels_top)
            cross_entropies_bottom = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits_bottom,
                                                                                    labels=labels_bottom)

            cross_entropies = cross_entropies_left + cross_entropies_right + cross_entropies_top + cross_entropies_bottom

            loss = tf.reduce_sum(processed_rewards * cross_entropies)

            global_step = tf.train.get_or_create_global_step()

            train_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

        with tf.name_scope('summaries'):
            rollout_reward = tf.placeholder(
                shape=(),
                dtype=tf.float32
            )

            # the weights to the hidden layer can be visualized
            hidden_weights = tf.trainable_variables()[0]
            for h in range(args.hidden_dim):
                slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
                image = tf.reshape(slice_, [1, ROWS, COLS, 1])
                tf.summary.image('hidden_{:04d}'.format(h), image)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
                tf.summary.scalar('{}_max'.format(var.op.name), tf.reduce_max(var))
                tf.summary.scalar('{}_min'.format(var.op.name), tf.reduce_min(var))
                
            tf.summary.scalar('rollout_reward', rollout_reward)
            tf.summary.scalar('loss', loss)

            merged = tf.summary.merge_all()

        print('Number of trainable variables: {}'.format(len(tf.trainable_variables())))

    inner_env = gym.make()
    # tf.agents helper to more easily track consecutive pairs of frames
    # env = FrameHistory(inner_env, past_indices=[0, 1], flatten=False)
    # tf.agents helper to automatically reset the environment
    env = AutoReset(inner_env)

    with tf.Session(graph=g) as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.output_dir)
            print('Restoring from {}'.format(restore_path))
            saver.restore(sess, restore_path)
        else:
            sess.run(init)

        summary_path = os.path.join(args.output_dir, 'summary')
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        # lowest possible score after an episode as the
        # starting value of the running reward
        _rollout_reward = -21.0

        for i in range(args.n_epoch):
            print('>>>>>>> epoch {}'.format(i+1))

            print('>>> Rollout phase')
            epoch_memory = []
            episode_memory = []

            # The loop for actions/steps
            _observation, _reward, _done, _ = env.step(action=[])
            while True:
                # sample one action with the given probability distribution
                _label = sess.run(sample_action, feed_dict={observations: [_observation]})

                _action = _label

                current_x, _reward, _done, _ = env.step(_action)

                if args.render:
                    env.render()
                
                # record experience
                episode_memory.append((_observation, _label, _reward))

                # # Get processed frame delta for the next step
                # pair_state = _pair_state
                #
                # current_state, previous_state = pair_state
                # current_x = current_state  # prepro(current_state)
                # previous_x = previous_state  # prepro(previous_state)
                #
                # _observation = current_x - previous_x

                _observation = current_x

                if _done:
                    obs, lbl, rwd = zip(*episode_memory)

                    # processed rewards
                    prwd = discount_rewards(rwd, args.gamma)
                    prwd -= np.mean(prwd)
                    prwd /= np.std(prwd)

                    # store the processed experience to memory
                    epoch_memory.extend(zip(obs, lbl, prwd))
                    
                    # calculate the running rollout reward
                    _rollout_reward = 0.9 * _rollout_reward + 0.1 * sum(rwd)

                    episode_memory = []

                    if args.render:
                        _ = input('episode done, press Enter to replay')
                        epoch_memory = []
                        continue

                if len(epoch_memory) >= ROLLOUT_SIZE:
                    break

            # add to the global memory
            MEMORY.extend(epoch_memory)

            print('>>> Train phase')
            print('rollout reward: {}'.format(_rollout_reward))

            # Here we train only once.
            _, _global_step = sess.run([train_op, global_step])

            if _global_step % args.save_checkpoint_steps == 0:

                print('Writing summary')

                feed_dict = {rollout_reward: _rollout_reward}
                summary = sess.run(merged, feed_dict=feed_dict)

                summary_writer.add_summary(summary, _global_step)

                save_path = os.path.join(args.output_dir, 'model.ckpt')
                save_path = saver.save(sess, save_path, global_step=_global_step)
                print('Model checkpoint saved: {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pong trainer')
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=6000)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=r'D:\temp\rl\pong')
    parser.add_argument(
        '--job-dir',
        type=str,
        default=r'D:\temp\rl\pong')

    parser.add_argument(
        '--restore',
        default=False,
        action='store_true')
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=1)

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-3)
    parser.add_argument(
        '--decay',
        type=float,
        default=0.99)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99)
    parser.add_argument(
        '--laziness',
        type=float,
        default=0.01)
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=200)

    args = parser.parse_args()

    # save all checkpoints
    args.max_to_keep = args.n_epoch // args.save_checkpoint_steps

    main(args)
