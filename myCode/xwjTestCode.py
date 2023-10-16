import sys
import numpy as np
import logging

# 地图长度常量，单位：米
MAP_LENGTH = 50
MAP_WIDTH = 50
# 机器人数量
ROBOT_NUM = 4
# 每秒帧数 50 FPS
FPS = 50
# 初始资金 200000
MONEY = 200000
# 机器人-工作台判定距离 0.4 米
ROBOT_WORKBENCH_DISTANCE = 0.4
# 机器人半径（常态） 0.45 米
ROBOT_RADIUS = 0.45
# 机器人半径（持有物品）
ROBOT_RADIUS_HOLDING = 0.53
# 机器人密度* 20 单位：kg/m2。质量=面积*密度
ROBOT_DENSITY = 20
# 最大前进速度* 6 米/s
ROBOT_MAX_FORWARD_SPEED = 6
# 最大后退速度* 2 米/s
ROBOT_MAX_BACKWARD_SPEED = 2
# 最大旋转速度* π/s rotate x [-π, π]
ROBOT_MAX_ROTATE_SPEED = np.pi
# 最大牵引力* 250 N 机器人的加速/减速/防侧滑均由牵引力驱动
ROBOT_MAX_TRACTION = 250
# 最大力矩* 50 N*m 机器人的旋转由力矩驱动
ROBOT_MAX_TORQUE = 50


class Map:
    def __init__(self):
        self.money = 0
        self.robots = []
        self.workbenches = []
        # 二维数组，存放障碍物的位置，0表示没有障碍物，1表示有障碍物
        self.obstacles = np.zeros((MAP_LENGTH * 2, MAP_WIDTH * 2))


class Robot:
    def __init__(self):
        self.id = -1
        # 信息：所处工作台ID 携带物品类型 时间价值系数 碰撞价值系数 角速度 线速度x 线速度y 朝向 坐标x 坐标y
        self.worktableID = -1
        self.goods_type = 0
        self.time_coefficient = 0.0
        self.collision_coefficient = 0.0
        self.angle_speed = 0.0
        self.line_speed = (0.0, 0.0)
        self.orientation = 0.0
        self.loc = (0, 0)

        self.on_task_dest = -1  # the worktable robot heading for 机器人的任务目标（以工作台的编号为目标），-1表示没有任务
        self.task = 0  # 1 buy, 2 sell, 3   机器人的任务：买、卖以及销毁携带的物品
        self.check_flag = False  # 检查机器人是否携带有物品：False是没有，True是有
        self.loc = (0, 0)  # 位置信息x,y
        self.goods_type_info = 0  # manual  把物品从机器人交给工作台的过渡，可以理解为机器人把物品给出去后的记录信息（记录刚刚给出去了哪个物品）
        self.optional_wt = []  # 根据机器人当前携带的物品，他能去的工作台列表

    def forward(self, line_speed):
        """
        forward: 机器人前进
        :param line_speed: 机器人前进的线速度
        :return:
        """
        sys.stdout.write('forward %d %f\n' % (self.id, line_speed))

    def rotate(self, angle_speed):
        """
        rotate: 机器人旋转
        :param angle_speed: 机器人旋转的角速度
        :return:
        """
        sys.stdout.write('rotate %d %f\n' % (self.id, angle_speed))

    def buy(self):
        """
        buy: 机器人买入
        :return:
        """
        sys.stdout.write('buy %d\n' % self.id)

    def sell(self):
        """
        sell: 机器人卖出
        :return:
        """
        sys.stdout.write('sell %d\n' % self.id)

    def destroy(self):
        """
        destroy: 机器人销毁
        :return:
        """
        sys.stdout.write('destroy %d\n' % self.id)


class Workbench:
    def __init__(self):
        self.id = -1
        # 信息：工作台类型 坐标x 坐标y 剩余生产时间（帧数） 原材料格状态(列表中材料编号) 产品格状态
        self.type = 0
        self.loc = (0, 0)
        self.remaining_time = 0
        self.raw_material_status = []
        self.production_status = 0

        self.raw_material = [-1]  # 原材料格，数组里面放原材料编号（1~7）
        self.work_cycle = 0  # 工作周期（帧数）固定的
        self.production = 0  # 产品编号
        self.material_ready = [0, 0, 0, 0, 0, 0, 0, 0]  # 某一格的原材料是否就绪？
        self.production_map_to_wt = []  # 生产的产品被哪些工作台所需要
        self.on_task_robot = -1  # 判断该工作台是否被机器人占用


def read_map():
    """
    read_map: 读取地图信息
    选手程序初始化时，将输入 100 行*100 列的字符组成的地图数据，然后紧接着一行 OK。
    :return:
    """
    map_info = Map()
    row = 0
    while True:
        col = 0
        line_information = input()
        if line_information == "OK":
            return map_info
        for char in line_information:
            if char == "#":
                map_info.obstacles[col][MAP_WIDTH * 2 - row - 1] = 1
            col += 1
        row += 1


def vec_to_loc(x, y):
    """
    :param x: 地图数组x下标
    :param y: 地图数组y下标
    :return: 返回地图坐标
    """
    return (x + 0.5) / 2, (y + 0.5) / 2


def convert_to_binary(number):
    """
    convert_to_binary: 将数字转化为二进制,提取二进制中为1的位数
    :param number: 数字
    :return: 二进制中为1的位数列表
    """
    binary_string = bin(number)
    positions = []
    for i, bit in enumerate(binary_string[::-1]):
        if bit == '1':
            positions.append(i)
    return positions


def read_first_frame():
    """
    read_first_frame: 读取第一帧，来新建控制台和机器人实体
    :return:
    """
    workbench_total = int(input())  # 工作台数量
    for workbench_id in range(workbench_total):
        line = input()
        parts = line.split(' ')  # 每行输入信息有了
        # 信息：工作台编号 坐标x 坐标y 剩余生产时间（帧数） 原材料格状态 产品格状态
        workbench = Workbench()
        workbench.id = workbench_id
        workbench.type = int(parts[0])
        workbench.loc = np.array([float(parts[1]), float(parts[2])])
        workbench.remaining_time = int(parts[3])
        workbench.raw_material_status = convert_to_binary(int(parts[4]))
        workbench.production_status = int(parts[5])
        map_info.workbenches.append(workbench)

    for robot_id in range(ROBOT_NUM):
        line = input()
        parts = line.split(' ')
        # 信息：所处工作台ID 携带物品类型 时间价值系数 碰撞价值系数 角速度 线速度x 线速度y 朝向 坐标x 坐标y
        robot = Robot()
        robot.id = robot_id
        robot.worktableID = int(parts[0])
        robot.goods_type = int(parts[1])
        robot.time_coefficient = float(parts[2])
        robot.collision_coefficient = float(parts[3])
        robot.angle_speed = float(parts[4])
        robot.line_speed = (float(parts[5]), float(parts[6]))
        robot.orientation = float(parts[7])
        robot.loc = np.array([float(parts[8]), float(parts[9])])
        map_info.robots.append(robot)
    line = input()
    if line == "OK":
        return


def read_util_ok():
    """
    read_util_ok: 读取每一帧信息，直到读到OK
    :return:
    """
    workbench_total = int(input())  # 工作台数量
    # 更新工作台信息
    for workbench_id in range(workbench_total):
        line = input()
        parts = line.split(' ')
        # 信息：剩余生产时间（帧数） 原材料格状态 产品格状态
        workbench = map_info.workbenches[workbench_id]
        workbench.remaining_time = int(parts[3])
        workbench.raw_material_status = convert_to_binary(int(parts[4]))
        workbench.production_status = int(parts[5])

    # 更新机器人信息
    for robot_id in range(ROBOT_NUM):
        line = input()
        parts = line.split(' ')
        # 信息：所处工作台ID 携带物品类型 时间价值系数 碰撞价值系数 角速度 线速度x 线速度y 朝向 坐标x 坐标y
        robot = map_info.robots[robot_id]
        robot.id = robot_id
        robot.worktableID = int(parts[0])
        robot.goods_type = int(parts[1])
        robot.time_coefficient = float(parts[2])
        robot.collision_coefficient = float(parts[3])
        robot.angle_speed = float(parts[4])
        robot.line_speed = (float(parts[5]), float(parts[6]))
        robot.orientation = float(parts[7])
        robot.loc = np.array([float(parts[8]), float(parts[9])])
    line = input()
    if line == "OK":
        return


def finish():
    """
    finish: 写入OK，并刷新缓冲区
    :return:
    """
    sys.stdout.write('OK\n')
    sys.stdout.flush()


def action():
    """
    action: 选手程序的主要逻辑
    :return:
    """
    # 最大前进速度前进
    map_info.robots[1].forward(ROBOT_MAX_FORWARD_SPEED)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    map_info = read_map()  # 读图
    finish()
    logging.debug("finish read map")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split(' ')
        frame_id = int(parts[0])  # 帧id
        map_info.money = int(parts[1])
        if frame_id == 1:
            read_first_frame()
            logging.debug("finish read first frame")
        else:
            read_util_ok()
            logging.debug("finish read frame")

        sys.stdout.write('%d\n' % frame_id)
        action()
        logging.debug("finish action")
        finish()
