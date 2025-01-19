
"""Packing task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p

MIN_VALID_SPACE = 0.05
OBS_CORRECTION = 0.05  # 观测修正值
FAIL_CORRECTION = 0.01  # 插入失败修正值
MAX_STEPS = 20
MARGIN = 0.01 #生成物块时，物块与容器边缘、物块与物块之间的最小距离

# 定义插入失败的原因枚举
class InsertFailReason:
    SUCCESS = 0  # 插入成功
    INVALID_BLOCK = 1  # block无效(如坐标顺序错误)
    OUT_OF_BOUNDS = 2  # block位置出界
    WIDTH_TOO_LARGE = 3  # block宽度过大
    LENGTH_TOO_LARGE = 4  # block长度过大
    NO_SPACE = 5  # 没有可用空间
    UNKNOWN = 6  # 其他未知原因insert

def split_space(original_space, block):
    # 提取原始空位和物块的坐标
    x0, y0, x1, y1 = original_space
    xb0, yb0, xb1, yb1 = block
    assert (x0<=xb0<xb1<=x1,y0<=yb0<yb1<=y1),"block out of space"
    # 初始化剩余区域列表
    remaining_spaces = []

    # 右侧区域
    if x1 - xb1 >= MIN_VALID_SPACE:
        remaining_spaces.append((xb1, y0, x1, y1))
    else:
        remaining_spaces.append(None)

    # 下方区域
    if y1 - yb1 >= MIN_VALID_SPACE:
        remaining_spaces.append((x0, yb1, x1, y1))
    else:
        remaining_spaces.append(None)
    # 左侧区域
    if xb0 - x0 >= MIN_VALID_SPACE:
        remaining_spaces.append((x0, y0, xb0, y1))
    else:
        remaining_spaces.append(None)
    # 上方区域
    if yb0 - y0 >= MIN_VALID_SPACE:
        remaining_spaces.append((x0, y0, x1, yb0))
    else:
        remaining_spaces.append(None)
    return remaining_spaces

class QuadTreeNode:
    def __init__(self, x0, y0, x1, y1, parent=None):
        self.x0, self.y0 = x0, y0  # 左上角坐标
        self.x1, self.y1 = x1, y1  # 右下角坐标
        self.children = None  # 四个子节点：左上、右上、左下、右下
        self.is_occupied = False  # 节点是否被完全占用
        self.parent = parent  # 父节点引用
        self.block_id = None  # 用于记录分割该节点的物块ID

    def update_occupied_status(self):
        """递归更新节点的占用状态"""
        if self.children is None:
            return self.is_occupied
        
        # 如果有子节点，检查所有子节点是否都被占用
        self.is_occupied = all(child is None or child.is_occupied for child in self.children)
        
        # 递归更新父节点
        if self.parent:
            self.parent.update_occupied_status()
        
        return self.is_occupied

    def insert(self, block, block_id):
        """
        将物块放入当前节点。如果成功放置，则分割节点。
        :param block: (x_b0, y_b0, x_b1, y_b1) 表示物块的范围
        :param block_id: 物块的唯一标识符
        :return: (是否成功放置, 失败原因)
        """
        x_b0, y_b0, x_b1, y_b1 = block

        # 检查物块坐标的有效性
        if not (x_b0 < x_b1 and y_b0 < y_b1):
            return False, InsertFailReason.INVALID_BLOCK

        # 检查物块是否在当前节点范围内
        if not (self.x0 <= x_b0 and x_b1 <= self.x1 and self.y0 <= y_b0 and y_b1 <= self.y1):
            # 进一步判断是宽度还是长度超出
            if x_b1 - x_b0 > self.x1 - self.x0:
                return False, InsertFailReason.WIDTH_TOO_LARGE
            elif y_b1 - y_b0 > self.y1 - self.y0:
                return False, InsertFailReason.LENGTH_TOO_LARGE
            print(f"x_b0:{x_b0},x_b1:{x_b1},self.x0:{self.x0},self.x1:{self.x1}")
            print(f"y_b0:{y_b0},y_b1:{y_b1},self.y0:{self.y0},self.y1:{self.y1}")
            return False, InsertFailReason.OUT_OF_BOUNDS

        # 如果是叶节点，则尝试分割
        if self.is_leaf():
            # 分割节点
            if self.subdivide(block, block_id):
                self.update_occupied_status()
                return True, InsertFailReason.SUCCESS
            else:
                return False, InsertFailReason.NO_SPACE

        # 如果不是叶节点，递归尝试放入子节点
        for child in self.children:
            if child:
                success, reason = child.insert(block, block_id)
                if success:
                    self.update_occupied_status()
                    return True, InsertFailReason.SUCCESS

        return False, InsertFailReason.NO_SPACE

    def subdivide(self, block, block_id):
        """
        按物块划分节点为上下左右四个子节点。
        :param block: (x_b0, y_b0, x_b1, y_b1) 表示物块的范围
        :param block_id: 物块的唯一标识符
        :return: 是否成功分割
        """
        x_b0, y_b0, x_b1, y_b1 = block

        # 检查物块尺寸是否超出当前节点范围
        if not (self.x0 <= x_b0 < x_b1 <= self.x1 and self.y0 <= y_b0 < y_b1 <= self.y1):
            return False

        remaining_space = split_space((self.x0,self.y0,self.x1,self.y1),block)
        # 初始化子节点列表
        children = []
        for space in remaining_space:
            if space is not None:
                x0, y0, x1, y1 = space
                child = QuadTreeNode(x0, y0, x1, y1, parent=self)
                children.append(child)
            else:
                children.append(None)

        # 更新当前节点的信息
        self.children = children
        self.block_id = block_id
        return True

    def contains(self, x, y):
        """检查点 (x, y) 是否在当前区域内"""
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def is_leaf(self):
        """检查当前节点是否是叶节点（无子节点）"""
        return self.children is None

class QuadTree:
    def __init__(self, x0, y0, x1, y1):
        self.root = QuadTreeNode(x0, y0, x1, y1)

    def query_available_space(self, block):
        """查询适合放置物块的空位"""
        def _query(node):
            if node.is_occupied:
                return None
            if node.is_leaf():
                return (node.x0, node.y0, node.x1, node.y1)
            for child in node.children:
                result = _query(child)
                if result:
                    return result
            return None

        return _query(self.root)

class PackingBoxes(Task):
  """Packing task."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = MAX_STEPS

  def reset(self, env):
    super().reset(env)

    # Add container box.
    container_x, container_y, container_z = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
    zone_size = (container_x, container_y, container_z)
    zone_pose = self.get_random_pose(env, zone_size)
    container_template = 'container/container-template.urdf'
    half = np.float32(zone_size) / 2
    replace = {'DIM': zone_size, 'HALF': half}
    container_urdf = self.fill_template(container_template, replace)
    env.add_object(container_urdf, zone_pose, 'fixed')
    os.remove(container_urdf)


    bboxes = []

    class TreeNode:

      def __init__(self, parent, children, bbox):
        self.parent = parent
        self.children = children
        self.bbox = bbox  # min x, min y, min z, max x, max y, max z

    def KDTree(node):
      size = node.bbox[3:] - node.bbox[:3]

      # Choose which axis to split.
      split = size > 2 * MIN_VALID_SPACE
      if np.sum(split) == 0:
        bboxes.append(node.bbox)
        return
      split = np.float32(split) / np.sum(split)
      split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

      # Split along chosen axis and create 2 children
      cut_ind = np.random.rand() * \
          (size[split_axis] - 2 * MIN_VALID_SPACE) + \
          node.bbox[split_axis] + MIN_VALID_SPACE
      child1_bbox = node.bbox.copy()
      child1_bbox[3 + split_axis] = cut_ind - MARGIN / 2.
      child2_bbox = node.bbox.copy()
      child2_bbox[split_axis] = cut_ind + MARGIN / 2.
      node.children = [
          TreeNode(node, [], bbox=child1_bbox),
          TreeNode(node, [], bbox=child2_bbox)
      ]
      KDTree(node.children[0])
      KDTree(node.children[1])

    # Split container space with KD trees.
    stack_size = np.array(zone_size)
    stack_size[0] -= MARGIN
    stack_size[1] -= MARGIN
    root_size = (MARGIN, MARGIN, 0) + tuple(stack_size)
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
      object_ids.append((box_id, (0, None)))
      icolor = np.random.choice(range(len(colors)), 1).squeeze()
      p.changeVisualShape(box_id, -1, rgbaColor=colors[icolor] + [1])
      object_points[box_id] = self.get_object_points(box_id)

    # Randomly select object in box and save ground truth pose.
    object_volumes = []
    true_poses = []
    # self.goal = {'places': {}, 'steps': []}
    for object_id, _ in object_ids:
      true_pose = p.getBasePositionAndOrientation(object_id)
      object_size = p.getVisualShapeData(object_id)[0][3]
      object_volumes.append(np.prod(np.array(object_size) * 100))
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

    self.goals.append((
        object_ids, np.eye(len(object_ids)), true_poses, False, True, 'zone',
        (object_points, [(zone_pose, zone_size)]), 1))
