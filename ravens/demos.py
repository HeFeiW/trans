# coding=utf-8
# Copyright 2024 The Ravens Authors.
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

"""Data collection script."""

import os

from absl import app
from absl import flags

import numpy as np

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import ContinuousEnvironment
from ravens.environments.environment import Environment

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'packing-with-error', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 3, '')
flags.DEFINE_bool('continuous', False, '')
flags.DEFINE_integer('steps_per_seg', 3, '')

FLAGS = flags.FLAGS


def main(unused_argv):

  # Initialize environment and task.
  env_cls = ContinuousEnvironment if FLAGS.continuous else Environment
  env = env_cls(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
  task = tasks.names[FLAGS.task](continuous=FLAGS.continuous)
  task.mode = FLAGS.mode

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))
    #debug
  print(f"Dataset_path:{os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}')}")
  print(f"n_episodes:{dataset.n_episodes}")
  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  # Determine max steps per episode.
  max_steps = task.max_steps
  if FLAGS.continuous:
    max_steps *= (FLAGS.steps_per_seg * agent.num_poses)
  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = None
    feedback = []
    reward = 0
    for _ in range(max_steps):
      act = agent.act(obs, info, feedback)
      episode.append((obs, act, reward, info, feedback))
      obs, reward, done, info, feedback = env.step(act)
      total_reward += reward
      print(f'Total Reward: {total_reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info, feedback))

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    # TODO whf for packing-with-error oracle is not optimal, 
    # set 0.70 as a rather good threshold
    if total_reward > 0.80:
      dataset.add(seed, episode)

if __name__ == '__main__':
  app.run(main)
