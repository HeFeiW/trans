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

"""Packing task."""

import os
import re
import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p

import collections
import os
import random
import string
import tempfile

import cv2
from ravens.tasks import cameras
from ravens.tasks import planners
from ravens.tasks import primitives
from ravens.tasks.grippers import Suction


import six

class PackingWithError1(Task):
  """Packing task."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 20

  def reset(self, env):
    super().reset(env)

    # Add container box.
    zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
    zone_pose = self.get_random_pose(env, zone_size)
    container_template = 'container/container-template.urdf'
    half = np.float32(zone_size) / 2
    replace = {'DIM': zone_size, 'HALF': half}
    container_urdf = self.fill_template(container_template, replace)
    env.add_object(container_urdf, zone_pose, 'fixed')
    os.remove(container_urdf)

    margin = 0.005
    min_object_dim = 0.05
    bboxes = []

    class TreeNode:

      def __init__(self, parent, children, bbox):
        self.parent = parent
        self.children = children
        self.bbox = bbox  # min x, min y, min z, max x, max y, max z

    def KDTree(node):
      size = node.bbox[3:] - node.bbox[:3]

      # Choose which axis to split.
      split = size > 2 * min_object_dim
      if np.sum(split) == 0:
        bboxes.append(node.bbox)
        return
      split = np.float32(split) / np.sum(split)#每个轴被选中的概率
      split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

      # Split along chosen axis and create 2 children
      cut_ind = np.random.rand() * \
          (size[split_axis] - 2 * min_object_dim) + \
          node.bbox[split_axis] + min_object_dim
      child1_bbox = node.bbox.copy()
      child1_bbox[3 + split_axis] = cut_ind - margin / 2.
      child2_bbox = node.bbox.copy()
      child2_bbox[split_axis] = cut_ind + margin / 2.
      node.children = [
          TreeNode(node, [], bbox=child1_bbox),
          TreeNode(node, [], bbox=child2_bbox)
      ]
      KDTree(node.children[0])
      KDTree(node.children[1])

    # Split container space with KD trees.
    stack_size = np.array(zone_size)
    stack_size[0] -= margin
    stack_size[1] -= margin
    root_size = (margin, margin, 0) + tuple(stack_size)
    root = TreeNode(None, [], bbox=np.array(root_size))
    KDTree(root)

    colors = [utils.COLORS[c] for c in utils.COLORS if c != 'brown']

    # Add objects in container.
    object_points = {}
    object_ids = []
    bboxes = np.array(bboxes)
    object_template = 'box/box-template.urdf'
    for bbox in bboxes:
      size = bbox[3:] - bbox[:3]
      position = size / 2. + bbox[:3]
      position[0] += -zone_size[0] / 2
      position[1] += -zone_size[1] / 2
      pose = (position, (0, 0, 0, 1))
      pose = utils.multiply(zone_pose, pose)
      urdf = self.fill_template(object_template, {'DIM': size})
      box_id = env.add_object(urdf, pose)
      os.remove(urdf)
      object_ids.append((box_id, (0, None)))#后面的tuple表示什么？
      icolor = np.random.choice(range(len(colors)), 1).squeeze()
      p.changeVisualShape(box_id, -1, rgbaColor=colors[icolor] + [1])
      object_points[box_id] = self.get_object_points(box_id)

    # Randomly select object in box and save ground truth pose.
    object_volumes = []
    true_poses = []
    object_sizes = []
    # self.goal = {'places': {}, 'steps': []}
    for object_id, _ in object_ids:
      true_pose = p.getBasePositionAndOrientation(object_id)
      object_size = p.getVisualShapeData(object_id)[0][3]
      object_volumes.append(np.prod(np.array(object_size) * 100))
      object_sizes.append(object_size)
      pose = self.get_random_pose(env, object_size)
      p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
      true_poses.append(true_pose)
      # self.goal['places'][object_id] = true_pose
      # symmetry = 0  # zone-evaluation: symmetry does not matter
      # self.goal['steps'].append({object_id: (symmetry, [object_id])})
    # self.total_rewards = 0
    # self.max_steps = len(self.goal['steps']) * 2

    # Sort oracle picking order by object size.
    # self.goal['steps'] = [
    #     self.goal['steps'][i] for i in
    #.    np.argsort(-1 * np.array(object_volumes))
    # ]
    # 生成匹配矩阵
    match_matrix = np.eye(len(object_ids))
    # 观测误差
    obs_error = 0.02
    # 物块两两比较obj_size,如果两个维度差值小于观测误差,则认为匹配;旋转90度后,如果两个维度差值小于观测误差,则认为匹配
    for i in range(len(object_ids)):
      for j in range(i+1,len(object_ids)):
        obj_size_i = object_sizes[i]
        obj_size_j = object_sizes[j]
        if abs(obj_size_i[0]-obj_size_j[0])<obs_error and abs(obj_size_i[1]-obj_size_j[1])<obs_error:
          match_matrix[i,j] = 1
          match_matrix[j,i] = 1
        if abs(obj_size_i[0]-obj_size_j[1])<obs_error and abs(obj_size_i[1]-obj_size_j[0])<obs_error:
          match_matrix[i,j] = 1
          match_matrix[j,i] = 1
    self.match_matrix = match_matrix
    self.goals.append((
        object_ids, np.eye(len(object_ids)), true_poses, False, True, 'zone',
        (object_points, [(zone_pose, zone_size)]), 1))
  def _discrete_oracle(self, env):
    """Discrete oracle agent."""
    OracleAgent = collections.namedtuple('OracleAgent', ['act'])

    def act(obs, info,feedback=None,last_act=None):  # pylint: disable=unused-argument
      """Calculate action."""
      print(f"feedback: {feedback}")
      obj_ids, _, targs, _, _, _, _, _ = self.goals[0]
      if feedback is not None:
        wrong_obj = feedback
        for i in range(len(obj_ids)):
          if obj_ids[i] == wrong_obj:
            wrong_obj_index = i
            break
        for i in range(len(targs)):
          if (targs[i][0] == last_act['pose1'][0]).all():
            print(f"targs[{i}][0]: {targs[i][0]}")
            print(f"last_act['pose1'][0]: {last_act['pose1'][0]}")
            self.match_matrix[i][wrong_obj_index] = 0
            self.match_matrix[wrong_obj_index][i] = 0
      print(f"self.match_matrix: {self.match_matrix}") 

      # Oracle uses perfect RGB-D orthographic images and segmentation masks.
      _, hmap, obj_mask = self.get_true_image(env)

      # Unpack next goal step.
      objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]
      # Match objects to targets without replacement.
      if not replace:

        # Modify a copy of the match matrix.
        matches = self.match_matrix.copy()

        # Ignore already matched objects.
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            if self.is_match(pose, targs[j], symmetry):
              matches[i, :] = 0
              matches[:, j] = 0

      # Get objects to be picked (prioritize farthest from nearest neighbor).
      nn_dists = []
      nn_targets = []
      for i in range(len(objs)):
        object_id, (symmetry, _) = objs[i]
        xyz, _ = p.getBasePositionAndOrientation(object_id)
        targets_i = np.argwhere(self.match_matrix[i, :]).reshape(-1)
        if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
          targets_xyz = np.float32([targs[j][0] for j in targets_i])
          dists = np.linalg.norm(
              targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
          nn = np.argmin(dists)
          nn_dists.append(dists[nn])
          nn_targets.append(targets_i[nn])

        # Handle ignored objects.
        else:
          nn_dists.append(0)
          nn_targets.append(-1)
      order = np.argsort(nn_dists)[::-1]#TODO 什么意思

      # Filter out matched objects.
      order = [i for i in order if nn_dists[i] > 0]

      pick_mask = None
      for pick_i in order:
        pick_mask = np.uint8(obj_mask == objs[pick_i][0])

        # Erode to avoid picking on edges.
        # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

        if np.sum(pick_mask) > 0:
          break

      # Trigger task reset if no object is visible.
      if pick_mask is None or np.sum(pick_mask) == 0:
        self.goals = []
        print('Object for pick is not visible. Skipping demonstration.')
        return 

      # Get picking pose.
      #whf added
      for config in env.agent_cams:
        env.save_image(config) 
      pick_prob = np.float32(pick_mask)
      pick_pix = utils.sample_distribution(pick_prob)#TODO 什么意思
      # For "deterministic" demonstrations on insertion-easy, use this:
      # pick_pix = (160,80)
      pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                  self.bounds, self.pix_size)
      pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

      # Get placing pose.
      targ_pose = targs[nn_targets[pick_i]]  # pylint: disable=undefined-loop-variable
      obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])  # pylint: disable=undefined-loop-variable
      if not self.sixdof:
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
        obj_pose = (obj_pose[0], obj_quat)
      world_to_pick = utils.invert(pick_pose)
      obj_to_pick = utils.multiply(world_to_pick, obj_pose)
      pick_to_obj = utils.invert(obj_to_pick)
      place_pose = utils.multiply(targ_pose, pick_to_obj)

      # Rotate end effector?
      if not rotations:
        place_pose = (place_pose[0], (0, 0, 0, 1))

      place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

      return {'pose0': pick_pose, 'pose1': place_pose}

    return OracleAgent(act)
  def reward(self):
    """Get delta rewards for current timestep.

    Returns:
      A tuple consisting of the scalar (delta) reward, plus `extras`
        dict which has extra task-dependent info from the process of
        computing rewards that gives us finer-grained details. Use
        `extras` for further data analysis.
    """
    reward, info = 0, {}

    if self.goals:
      # Unpack next goal step.
      objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]

      # Evaluate by matching object poses.
      if metric == 'pose':
        step_reward = 0
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            target_pose = targs[j]
            if self.is_match(pose, target_pose, symmetry):
              step_reward += max_reward / len(objs)
              break

      # Evaluate by measuring object intersection with zone.
      elif metric == 'zone':
        zone_pts, total_pts = 0, 0
        obj_pts, zones = params
        for zone_pose, zone_size in zones:

          # Count valid points in zone.
          for obj_id in obj_pts:
            pts = obj_pts[obj_id]
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            world_to_zone = utils.invert(zone_pose)
            obj_to_zone = utils.multiply(world_to_zone, obj_pose)
            pts = np.float32(utils.apply(obj_to_zone, pts))
            if len(zone_size) > 1:
              valid_pts = np.logical_and.reduce([
                  pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
                  pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
                  pts[2, :] < zone_size[2]])

            zone_pts += np.sum(np.float32(valid_pts))
            total_pts += pts.shape[1]
        step_reward = max_reward * (zone_pts / total_pts)

      # Get cumulative rewards and return delta.
      reward = self.progress + step_reward - self._rewards
      self._rewards = self.progress + step_reward

      # Move to next goal step if current goal step is complete.
      if np.abs(max_reward - step_reward) < 0.01:
        self.progress += max_reward  # Update task progress.
        self.goals.pop(0)

    else:
      # At this point we are done with the task but executing the last movements
      # in the plan. We should return 0 reward to prevent the total reward from
      # exceeding 1.0.
      reward = 0.0

    return reward, info
  def feedback(self):
    """检测环境中物体之间的碰撞。
    
    返回:
        bool: True表示有碰撞发生，False表示没有碰撞
    """
    # 获取所有物体的ID
    if len(self.goals) == 0:
      return False
    objs, _, targs, _, _, _, _, _ = self.goals[0]
    object_ids = [obj[0] for obj in objs]
    
    # 检查每对物体之间的碰撞
    for i in range(len(object_ids)):
        for j in range(i+1, len(object_ids)):
            # 使用getClosestPoints检测碰撞，距离阈值为0表示接触
            closest_points = p.getClosestPoints(
                object_ids[i], 
                object_ids[j], 
                distance=0.0
            )
            # 如果有接触点，说明发生碰撞
            if len(closest_points) > 0:
                #返回两个物体中z值较大的那个
                if p.getBasePositionAndOrientation(object_ids[i])[0][2] > p.getBasePositionAndOrientation(object_ids[j])[0][2]:
                  return object_ids[i]
                else:
                  return object_ids[j]
                
    # 如果没有检测到任何碰撞
    return None

  
