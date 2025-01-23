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
          match_matrix[i,j] = -1
          match_matrix[j,i] = -1
    self.match_matrix = match_matrix
    self.goals.append((
        object_ids, match_matrix, true_poses, False, True, 'zone',
        (object_points, [(zone_pose, zone_size)]), 1))
    self.home_occupied = np.zeros(len(object_ids))
  def feedback(self, obj_poses):
    def is_laid_flat(obj_pose, tolerance=0.1):
      """
      通过检查上表面法向量判断物块是否平放
      Args:
          quaternion: (x,y,z,w) 格式的四元数
          tolerance: 允许的误差范围
      Returns:
          bool: 是否平放
      """
      # 获取旋转矩阵
      quaternion = obj_pose[1]
      rot_matrix = p.getMatrixFromQuaternion(quaternion)
      rot_matrix = np.array(rot_matrix).reshape(3, 3)
      
      # 上表面法向量（通常是旋转矩阵的第三列）
      up_vector = rot_matrix[:, 2]
      
      # 计算与竖直方向(0,0,1)的夹角余弦值
      cos_angle = abs(np.dot(up_vector, np.array([0, 0, 1])))
      
      # 如果余弦值接近1或-1，说明物块平放
      return cos_angle > (1 - tolerance)
    def is_not_on_top_of_others(obj_pose, tolerance=0.1):
      return not obj_pose[0][2] > 0.06
    def is_aligned_with_container(obj_id, tolerance=0.2618):
      container_id = self.env.obj_ids["fixed"][0]
      container_pose = p.getBasePositionAndOrientation(container_id)
      contact_points = p.getContactPoints(container_id, obj_id)
      if len(contact_points) >0:
        return True
      container_rotation_matrix = np.array(p.getMatrixFromQuaternion(container_pose[1])).reshape(3, 3)
      obj_pose = p.getBasePositionAndOrientation(obj_id)
      obj_rotation_matrix = np.array(p.getMatrixFromQuaternion(obj_pose[1])).reshape(3, 3)
      angle_diff = np.arccos(np.clip(np.dot(container_rotation_matrix[:, 0], obj_rotation_matrix[:, 0]), -1.0, 1.0))
      angle_diff_90 = angle_diff-np.pi/2
      return min(angle_diff,angle_diff_90) < tolerance
    laid_flat = [{'obj_id': obj_pose['obj_id'], 'is_laid_flat': is_laid_flat(obj_pose['pose'])} for obj_pose in obj_poses]
    on_top_of_others = [{'obj_id': obj_pose['obj_id'], 'is_on_top_of_others': is_not_on_top_of_others(obj_pose['pose'])} for obj_pose in obj_poses]
    aligned_with_container = [{'obj_id': obj_pose['obj_id'], 'is_aligned_with_container': is_aligned_with_container(obj_pose['obj_id'])} for obj_pose in obj_poses]
    if np.sum([laid_flat['is_laid_flat'] for laid_flat in laid_flat]) == len(laid_flat) and np.sum([on_top_of_others['is_on_top_of_others'] for on_top_of_others in on_top_of_others]) == len(on_top_of_others):
      return f"success"
    else:
      feedback = ""
      for i in range(len(laid_flat)):
        if not laid_flat[i]['is_laid_flat']:
          feedback += f"object {laid_flat[i]['obj_id']} is not properly placed in the container\n"
        if not on_top_of_others[i]['is_on_top_of_others']:
          feedback += f"object {on_top_of_others[i]['obj_id']} is not on top of others\n"
        if not aligned_with_container[i]['is_aligned_with_container']:
          feedback += f"object {aligned_with_container[i]['obj_id']} is not aligned with the container\n"
      return feedback
  def _discrete_oracle(self, env):
    """Discrete oracle agent."""
    OracleAgent = collections.namedtuple('OracleAgent', ['act'])
    self.env = env
    def act(obs, info,feedback="success"):  # pylint: disable=unused-argument
      """Calculate action."""
      # 解析feedback
      feedback = str(feedback)
      print(f"feedback: {feedback}")
      #如果feedback包含success,则把当前的位置原住民的home_occupied设置为1
      
      obj_to_correct = int(re.search(r'\[(\d+)', feedback).group(1)) if re.search(r'\[(\d+)', feedback) else None

      # Oracle uses perfect RGB-D orthographic images and segmentation masks.
      _, hmap, obj_mask = self.get_true_image(env)

      # Unpack next goal step.
      objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]
      # Match objects to targets without replacement.
      if not replace:

        # Modify a copy of the match matrix.
        matches = matches.copy()

        # Ignore already matched objects.
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            if self.is_match(pose, targs[j], symmetry):
              matches[i, :] = 0
              matches[:, j] = 0
      if "success" in feedback:
        #遍历所有target,如果该位置有物体,则把该物体的home_occupied设置为1
        for i in range(len(self.home_occupied)):
          ray_from_point = (targs[i][0][0],targs[i][0][1],targs[i][0][2]-0.02)
          ray_to_point = (targs[i][0][0],targs[i][0][1],targs[i][0][2]-0.04)
          ray_result = p.rayTest(rayFromPosition=ray_from_point, rayToPosition=ray_to_point, physicsClientId=self.env.client)
          if len(ray_result) > 0:
            self.home_occupied[i] = 1
        print(f"object{np.argwhere(self.home_occupied).reshape(-1)}'s home is occupied")
          
      if obj_to_correct is not None:
        for i in range(len(objs)):
          if objs[i][0] == obj_to_correct:
            obj_to_correct_index = i
            obj_to_correct_pose_init = p.getBasePositionAndOrientation(objs[i][0])
            obj_to_correct_pose = (obj_to_correct_pose_init[0],obj_to_correct_pose_init[1],0.5)
            break
        # 获得原本应该在这个位置上的物体
        for i in range(len(objs)):
          if self.is_match(obj_to_correct_pose, targs[i], symmetry):
            aborigine = i
            self.match_matrix[aborigine,obj_to_correct_index] = 0
            self.match_matrix[obj_to_correct_index,aborigine] = 0
            break
      
        for i in range(len(objs)):
          obj_pose = p.getBasePositionAndOrientation(objs[i][0])
      # Get objects to be picked (prioritize farthest from nearest neighbor).
      nn_dists = []
      nn_targets = []
      for i in range(len(objs)):
        object_id, (symmetry, _) = objs[i]
        xyz, orientation = p.getBasePositionAndOrientation(object_id)
        targets_i = np.argwhere(self.match_matrix[i, :]).reshape(-1)
        targets_xyz = []
        if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
          targs_i_available = []
          for j in targets_i:
            if self.home_occupied[j] == 1:
                pass
            else:
                targs_i_available.append(j)
                targets_xyz.append(targs[j][0])
          if len(targets_xyz) == 0:
            targets_xyz.append(targs[i][0])
            targs_i_available.append(i)
          dists = np.linalg.norm(
              targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
          nn = np.argmin(dists)
          nn_dists.append(dists[nn])
          nn_targets.append(targs_i_available[nn])
        # Handle ignored objects.
        else:
          nn_dists.append(0)
          nn_targets.append(-1)
      order = np.argsort(nn_dists)[::-1]
      print(f"order: {order}")
      if obj_to_correct is not None:
        print(f"obj_to_correct_index: {obj_to_correct_index}")
        order_copy = order.copy()
        order = []
        order.append(obj_to_correct_index)
        for i in range(len(order_copy)):
          if order_copy[i] != obj_to_correct_index:
            order.append(order_copy[i])
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
      pick_prob = np.float32(pick_mask)
      pick_pix = utils.sample_distribution(pick_prob)#TODO 什么意思
      # For "deterministic" demonstrations on insertion-easy, use this:
      # pick_pix = (160,80)
      pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                  self.bounds, self.pix_size)
      # 如果match_matrix中该位置为-1,则将物块旋转90度
      pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))
      if self.match_matrix[pick_i,pick_i] == -1:
        pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, np.pi/2, 1)))

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