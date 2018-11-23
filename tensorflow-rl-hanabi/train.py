"""
In most basic trial, the agent needs to understand to select actions 0 -> 1 -> 2.

So rules are...
The agent has the cards: [1, 2, 3] x infinity
Each turn the agent selects which card to put down. It must start with '1'.
Once a '1' is on the table, he can put a '2' and after that a '3'
The agent can make up to 3 errors.

Rewards:
Max reward is 3.
For each error the agent gets -1 so min error is -3.
"""


import os
import argparse
from collections import deque

import numpy as np
import tensorflow as tf
from agents.tools.wrappers import AutoReset

from entities.game import Game
from helpers import discount_rewards


# Actions to select to drop any one of the cards
ACTIONS = list(range(0, Game.NUM_ACTIONS))

# Observation includes entry for each card type which contains '0' or '1' if the card is on the table.
OBSERVATION_DIM = len(Game.COLORS) * Game.CARDS_MAX_VALUE

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
        net = observations
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        logits = tf.layers.dense(net, len(ACTIONS), use_bias=False)

    return logits


def train(args):
    args_dict = vars(args)
    print('args: {}'.format(args_dict))

    with tf.Graph().as_default() as g:

        # rollout subgraph
        with tf.name_scope('rollout'):
            observations = tf.placeholder(shape=(None, OBSERVATION_DIM), dtype=tf.float32)

            logits = build_graph(observations)

            logits_for_sampling = tf.reshape(logits, shape=(1, len(ACTIONS)))

            # Sample the action to be played during rollout.
            sample_action = tf.multinomial(logits=logits_for_sampling, num_samples=1)
            sample_action = tf.squeeze(sample_action)

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

            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_logits,
                labels=labels
            )

            # Extra loss when the paddle is moved, to encourage more natural moves.
            probs = tf.nn.softmax(logits=train_logits)

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

            # # the weights to the hidden layer can be visualized
            # hidden_weights = tf.trainable_variables()[0]
            # for h in range(args.hidden_dim):
            #     slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
            #     image = tf.reshape(slice_, [1, 80, 80, 1])
            #     tf.summary.image('hidden_{:04d}'.format(h), image)

            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)
            #     tf.summary.scalar('{}_max'.format(var.op.name), tf.reduce_max(var))
            #     tf.summary.scalar('{}_min'.format(var.op.name), tf.reduce_min(var))

            tf.summary.scalar('rollout_reward', rollout_reward)
            tf.summary.scalar('loss', loss)

            merged = tf.summary.merge_all()

    game = Game()
    # tf.agents helper to more easily track consecutive pairs of frames
    # env = FrameHistory(inner_env, past_indices=[0, 1], flatten=False)
    # tf.agents helper to automatically reset the environment
    env = AutoReset(game)

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
        _rollout_reward = game.lowest_reward

        for i in range(args.n_epoch):
            print('>>>>>>> epoch {}'.format(i + 1))

            print('>>> Rollout phase')
            epoch_memory = []
            episode_memory = []

            # The loop for actions/steps
            _observation = np.zeros(OBSERVATION_DIM)
            while True:
                # sample one action with the given probability distribution
                _label = sess.run(sample_action, feed_dict={observations: [_observation]})

                _action = ACTIONS[_label]

                _observation, _reward, _done, _ = env.step(_action)
                _observation = _observation.reshape(OBSERVATION_DIM)

                if args.render:
                    env.render()

                # record experience
                episode_memory.append((_observation, _label, _reward))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epoch', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--output-dir', type=str, default=r'D:\temp\rl\hanabi')
    parser.add_argument('--job-dir', type=str, default=r'D:\temp\rl\hanabi')

    parser.add_argument('--restore', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--save-checkpoint-steps', type=int, default=1)

    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)

    _args = parser.parse_args()

    _args.max_to_keep = 5#_args.n_epoch // _args.save_checkpoint_steps

    train(_args)
