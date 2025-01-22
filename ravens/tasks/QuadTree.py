import numpy as np

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
