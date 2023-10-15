import sys
import numpy as np


# Map()用于存储地图信息
class Map():
    def __init__(self):
        self.width = 50  # 地图长度
        self.height = 50  # 地图宽度
        # 所以一行一列的长度是0.5
        self.money = 0  # 资金数
        self.worktable_num = 0  # 工作台数量
        self.robots = []  # 机器人实体数组
        self.worktables_list = []  # by order工作台实体数组：按顺序编号
        # 每种类别的物品当前时刻可以添加到指定工作台的编号，如1:[2,4,6]表示编号2、4、6号工作台当前原材料需要1号物品
        self.wt_to_be_added = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        # 哪些原材料工作台（类型1-3）已生产出成品（1,2,3）了，如[2,4,6]表示编号第2、4、6号工作台当前已生产出原材料（1-3号的一种）
        self.wt_to_be_purchased_123 = []
        # 哪些加工工作台（类型4-7）已生产出成品（4,5,6,7）了，值含义同上
        self.wt_to_be_purchased_4567 = []
        # 地图障碍物和工作台位置信息，用一个二维数组表示，每个元素是一个二维数组，表示障碍物或工作台的位置
        self.obstacles = []

    # def reset(self):
    #    self.wt_to_be_purchased = []
    #    self.wt_to_be_added = []


# Robot()用于存储机器人信息
class Robot():
    def __init__(self):
        self.id = -1  # id
        # 机器人半径为0.45，携带物品时半径为0.53，密度为20，质量为0.45*0.45*20*3.14=12.75
        # 所以直径为0.9，携带物品时直径为1.06，当机器人没有携带物品时，可以通过距离为1的通道，当机器人携带物品时，无法通过1的通道
        # self.radius = 0.45
        # self.radius_with_goods = 0.53
        # self.density = 20
        # self.quality = np.pi * np.square(self.radius) * self.density
        # self.quality_with_goods = np.pi * \
        #     np.square(self.radius_with_goods) * self.density
        self.on_task_dest = -1  # the worktable robot heading for 机器人的任务目标（以工作台的编号为目标），-1表示没有任务
        self.task = 0  # 1 buy, 2 sell, 3   机器人的任务：买、卖以及销毁携带的物品
        self.check_flag = False  # 检查机器人是否携带有物品：False是没有，True是有

        self.loc = (0, 0)  # 位置信息x,y
        self.worktableID = -1
        self.goods_type_info = 0  # manual  把物品从机器人交给工作台的过渡，可以理解为机器人把物品给出去后的记录信息（记录刚刚给出去了哪个物品）
        self.goods_type = 0  # [0, 7]   #机器人所携带的物品类型（0~7），0代表没有物品
        self.optional_wt = []  # 根据机器人当前携带的物品，他能去的工作台列表
        # self.time_coef = 0.0
        # self.collision_coef = 0.0
        self.angle_speed = 0.0  # 角速度
        self.line_speed = (0.0, 0.0)  # 线速度
        self.orientation = 0.0  # [-pi, pi] #方位角

    # The robot should go to the corresponding worktable(s) when it carries goods
    # 机器人在携带物品时，它能走向的对应工作台列表
    def get_optional_wt(self):
        ori_material = [0, 1, 2, 3, 4, 5, 6, 7]  # 能够携带的材料列表
        optional_worktables = [
            [], [4, 5, 9], [4, 6, 9], [5, 6, 9], [7, 9], [7, 9], [7, 9], [8, 9]
        ]  # 每种原材料分别有哪些工作台在使用（原材料序号从下标0开始），一共有7种原材料：例如1号原材料被4、5、9三个工作台需要
        self.optional_wt = optional_worktables[ori_material[self.goods_type]]

    # 机器人的行为：
    # forward: 机器人前进
    # 9999表示最大速度，-9999表示最小速度，0表示停止
    def forward(self, line_speed):
        sys.stdout.write('forward %d %f\n' % (self.id, line_speed))

    # rotate: 机器人旋转
    def rotate(self, angle_speed):
        sys.stdout.write('rotate %d %f\n' % (self.id, angle_speed))

    # buy: 机器人购买
    def buy(self):
        sys.stdout.write('buy %d\n' % self.id)

    # sell: 机器人卖出
    def sell(self):
        sys.stdout.write('sell %d\n' % self.id)

    # destroy: 机器人销毁
    def destroy(self):
        sys.stdout.write('destroy %d\n' % self.id)


class Worktable():
    # 初始化工作台
    def __init__(self, worktable_type):
        self.type = worktable_type  # 工作台类型（1~9）
        self.id = -1
        self.id_sell_after_buy = -1  # 卖给下一个工作台的id
        self.loc = (0, 0)  # 坐标
        self.raw_material = [-1]  # 原材料格，数组里面放原材料编号（1~7）
        self.work_cycle = 0  # 工作周期（帧数）固定的
        self.production = 0  # 产品编号
        self.remaining_time = 0  # 剩余生产时间（帧数）
        self.raw_material_status = '00000000'  # 原材料格的状态（二进制表示）
        self.material_ready = [0, 0, 0, 0, 0, 0, 0, 0]  # 某一格的原材料是否就绪？
        self.production_status = 0  # 产品格内是否放了产品（0没有，1有）
        self.production_map_to_wt = []  # 生产的产品被哪些工作台所需要
        self.on_task_robot = -1  # judge whether the worktable is occupied  判断该工作台是否被机器人占用
        # 方法
        self.get_info(worktable_type)  # 获得每个工作台的信息

    # 获得每个工作台的信息
    def get_info(self, type):
        info = [  # 9个工作台
            [[-1], 50, 1],  # 信息：原材料编号   工作周期    生产物品编号
            [[-1], 50, 2],
            [[-1], 50, 3],
            [[1, 2], 500, 4],
            [[1, 3], 500, 5],
            [[2, 3], 500, 6],
            [[4, 5, 6], 1000, 7],
            [[7], 1, 0],
            [[1, 2, 3, 4, 5, 6, 7], 1, 0]
        ]
        materials_map_to_wt = [  # 每种原材料分别有哪些工作台在使用（原材料序号从下标0开始），7种原材料：例如1号原材料被4、5、9三个工作台需要
            [], [4, 5, 9], [4, 6, 9], [5, 6, 9], [7, 9], [7, 9], [7, 9], [8, 9]
        ]
        self.raw_material = info[type - 1][0]  # 类型type的工作台的原材料编号
        self.work_cycle = info[type - 1][1]  # 其工作周期
        self.production = info[type - 1][2]  # 生产物品编号
        self.production_map_to_wt = materials_map_to_wt[self.production]  # 生产的产品被哪些工作台所需要

    # 机器人从该工作台购买产品，并找到最近的可使用的工作台卖掉
    def to_be_purchased(self):
        if self.production_status:  # 如果该工作台有产品了：
            if len(map_info.wt_to_be_added[self.production]):  # 如果该工作台生产出的产品，能被其他工作台利用：
                min_dist_wt = float('inf')  # 定义到目标工作台的距离为无限远
                for wt_id in map_info.wt_to_be_added[self.production]:  # 遍历利用该产品的工作台，目标是找到距离最近的那个工作台：
                    wt = map_info.worktables_list[wt_id]  # 拿到该工作台实体
                    if wt.material_ready[self.production] == 0:  # 如果该工作台的原材料格没有原材料，说明可以购买：
                        tmp_dist, _ = dist(self.loc, wt.loc)
                        if tmp_dist < min_dist_wt:
                            min_dist_wt = tmp_dist  # 更新到目标工作台的距离
                            min_id = wt_id
                if min_dist_wt != float('inf'):  # 找到最近的工作台了：
                    self.id_sell_after_buy = min_id
                    map_info.worktables_list[min_id].material_ready[self.production] = 1  # 让下一个工作台的原材料变成就绪状态
                    return True
        return False

    # 维护wt_to_be_add表，把空闲工作台加到原材料数组中
    def to_be_added(self):
        if self.production_status and self.remaining_time > 0:
            return False
        else:
            for i in range(1, 8):
                if i in self.raw_material and self.raw_material_status[i] == '0':
                    if self.id not in map_info.wt_to_be_added[i]:
                        map_info.wt_to_be_added[i].append(self.id)
            return True


# 我到时候，可能需要把障碍物的位置也得记录一下，需要维护一个二维数组，记录障碍物和工作台位置
def read_map() -> Map:  # 初始化地图信息
    """
    :return: 地图信息
    """
    map_information = Map()
    row = 0
    worktable_num = 0
    robot_count = 0
    while True:
        col = 0
        line_information = input()
        if line_information == "OK":
            map_information.worktable_num = worktable_num
            return map_information
        for char in line_information:
            # 如果是1~9，说明是工作台，那么就创建工作台实体，然后把工作台实体添加到工作台实体数组中
            if "1" <= char <= "9":  # 工作台
                worktable = Worktable(int(char))
                worktable.id = worktable_num  # 根据遇到的顺序给编号
                worktable.loc = np.array([0.25 + col * 0.5, 49.75 - row * 0.5])  # 算工作台中心位置坐标
                map_information.worktables_list.append(worktable)  # 添加工作台实体
                worktable_num += 1
            # 如果是A，说明是机器人，那么就创建机器人实体，然后把机器人实体添加到机器人实体数组中
            elif char == "A":  # 机器人
                robot = Robot()
                robot.id = robot_count
                robot.loc = np.array([0.25 + col * 0.5, 49.75 - row * 0.5])
                map_information.robots.append(robot)
                robot_count += 1
                # 如果是#，说明是障碍物，那么就把障碍物的位置添加到障碍物数组中
            elif char == "#":
                # np.array([0.25 + col * 0.5, 49.75 - row * 0.5])表示障碍物的中心位置坐标
                map_information.obstacles.append(
                    np.array([0.25 + col * 0.5, 49.75 - row * 0.5]))
            col += 1
        row += 1


# 获取工作台和机器人信息
def read_util_ok():
    while True:
        worktable_total = int(input())  # 输入工作台总数，这是下面几行的输入
        for worktable_id in range(worktable_total):  # 遍历每一个工作台：
            line = input()
            parts = line.split(' ')  # 每行输入信息有了
            # 信息：工作台编号   坐标x  坐标y    【剩余生产时间（帧数）  原材料格状态  产品格状态】
            map_info.worktables_list[worktable_id].remaining_time = int(
                parts[3])  # 写入每一个工作台的剩余生产时间
            map_info.worktables_list[worktable_id].raw_material_status = bin(int(parts[4]))[::-1][:-2].ljust(8,
                                                                                                             "0")  # 拆分二进制数，写入原材料状态
            map_info.worktables_list[worktable_id].production_status = int(
                parts[5])
            # find free worktables
            # 找空闲工作台
            if map_info.worktables_list[worktable_id].on_task_robot == -1:  # 如果工作台未被占用
                if parts[0] not in ['1', '2', '3']:  # 且是加工工作台
                    map_info.worktables_list[worktable_id].to_be_added()
                if parts[0] in ['1', '2', '3']:
                    if worktable_id not in map_info.wt_to_be_purchased_123:
                        if map_info.worktables_list[worktable_id].to_be_purchased():
                            map_info.wt_to_be_purchased_123.append(worktable_id)
                elif parts[0] in ['4', '5', '6', '7']:
                    if worktable_id not in map_info.wt_to_be_purchased_4567:
                        if map_info.worktables_list[worktable_id].to_be_purchased():
                            map_info.wt_to_be_purchased_4567.append(worktable_id)
        # 遍历每一个机器人
        for robot_id in range(4):
            line = input()
            parts = line.split(' ')
            # 信息：所处工作台ID    携带物品类型   时间价值系数     碰撞价值系数      角速度     线速度x    线速度y     朝向      坐标x     坐标y
            map_info.robots[robot_id].worktableID = int(parts[0])
            map_info.robots[robot_id].goods_type = int(parts[1])
            map_info.robots[robot_id].time_coef = float(parts[2])
            map_info.robots[robot_id].collision_coef = float(parts[3])
            map_info.robots[robot_id].angle_speed = float(parts[4])
            map_info.robots[robot_id].line_speed = (
                float(parts[5]), float(parts[6]))
            map_info.robots[robot_id].orientation = float(parts[7])
            map_info.robots[robot_id].loc = np.array(
                [float(parts[8]), float(parts[9])])
        line = input()
        if line == "OK":
            break


# 输出OK 表示初始化结束
def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()


# 机器人前进
def dist(ori, dest):
    """
    :param ori: 机器人的坐标
    :param dest: 目标坐标
    :return: 机器人到目标的距离，机器人到目标的方位角
    """
    dist = np.sqrt(np.sum(np.square(dest - ori)))
    delta_x = dest[0] - ori[0]
    delta_y = dest[1] - ori[1]
    if delta_x * delta_y == 0:
        if delta_x:
            ori_to_dest_radian = 0 if delta_x > 0 else np.pi
        elif delta_y:
            ori_to_dest_radian = np.pi / 2 if delta_y > 0 else -np.pi / 2
        else:
            ori_to_dest_radian = 0
    elif delta_x > 0:
        ori_to_dest_radian = np.arctan(delta_y / delta_x)
    elif delta_x < 0:
        tmp_radian = np.arctan(delta_y / delta_x)
        ori_to_dest_radian = tmp_radian + np.pi if tmp_radian < 0 else tmp_radian - np.pi
    return dist, ori_to_dest_radian


# 改变特定机器人的任务——如果是买入，把它的任务状态变成卖给下一个工作台，然后退出；如果是卖出，那么重置机器人状态，变成初始状态。
def reset(robot_id):
    # 如果当前机器人的状态是购买，说明在某处工作台已经拿到产品了
    if map_info.robots[robot_id].task == 1:
        tmp_wt_id = map_info.robots[robot_id].on_task_dest  # 存一下机器人当前任务目标（工作台的编号）
        map_info.robots[robot_id].on_task_dest = \
            map_info.worktables_list[map_info.robots[robot_id].on_task_dest] \
                .id_sell_after_buy  # 让机器人的任务目标从当前工作台变成下一个工作台（必须满足下一个sell_after_buy工作台的原材料是当前工作台的产品）
        # 对原来工作台的处理：
        map_info.worktables_list[
            tmp_wt_id].id_sell_after_buy = -1  # 那么对于原先机器人的目标工作台，由于产品被买走了，它的下一个sell_after_buy工作台就变成-1
        map_info.worktables_list[tmp_wt_id].on_task_robot = -1  # 它也不再被机器人占用
        # 对下一个sell_after_buy工作台的处理：
        map_info.worktables_list[map_info.robots[robot_id].on_task_dest].on_task_robot = robot_id  # 工作台被机器人占用
        # 机器人信息更新：
        map_info.robots[robot_id].task = 2  # 机器人任务变成卖出
        map_info.robots[robot_id].goods_type_info = map_info.robots[robot_id].goods_type  # 记住当前买入产品的产品信息
        map_info.robots[robot_id].check_flag = False  # 机器人不再有物品
        return

    # 如果当前机器人的任务是卖出，说明在目标工作台机器人已经把物品卖掉了
    if map_info.robots[robot_id].task == 2:
        map_info.worktables_list[map_info.robots[robot_id].on_task_dest].on_task_robot = -1  # 此工作台不再被占用
        map_info.worktables_list[map_info.robots[robot_id].on_task_dest].material_ready[
            map_info.robots[robot_id].goods_type_info] = 0  # 把目标工作台的原材料状态变成就绪【问题：为什么变成0，而不是变成1】
        map_info.wt_to_be_added[map_info.robots[robot_id].goods_type_info].remove(
            map_info.robots[robot_id].on_task_dest)  # 已经把产品卖给对应工作台了，该工作台不能再接受同类型的产品了，因此wt_to_be_added表格要把该工作台删掉

    # 卖出收尾工作结束，或者已经销毁了物品，现在对机器人的状态进行重置：
    map_info.robots[robot_id].goods_type_info = 0  # 忘记刚刚给出的信息
    map_info.robots[robot_id].on_task_dest = -1  # 机器人任务目标重置
    map_info.robots[robot_id].task = 0  # 机器人任务类型重置
    map_info.robots[robot_id].check_flag = False  # 机器人不再携带物品


# 检查机器人携带物品和目标工作台的产品信息是否匹配，不匹配则销毁
def check(robot_id):
    # 如果当前机器人有任务目标（要去的工作台编号）
    if map_info.robots[robot_id].on_task_dest > -1:  # 0~8
        # 如果当前机器人的任务是买入
        if map_info.robots[robot_id].task == 1:  # buy
            # 如果机器人当前携带的产品类型与下一个工作台需要的产品类型匹配
            if map_info.robots[robot_id].goods_type == map_info.worktables_list[
                map_info.robots[robot_id].on_task_dest].type:
                reset(robot_id)  # 设置机器人的任务状态
                return True
            # 如果不匹配，销毁之
            else:
                map_info.robots[robot_id].task = 3
                return False
        # 如果当前机器人的任务是卖出
        elif map_info.robots[robot_id].task == 2:  # sell
            if map_info.worktables_list[map_info.robots[robot_id].on_task_dest].raw_material_status[
                map_info.robots[robot_id].goods_type] == '0':  # 就是机器人要去的工作台的特定原材料还是空的，那么就设置机器人信息，否则销毁产品
                reset(robot_id)
            else:
                map_info.robots[robot_id].task = 3
            return False


# 碰撞检测
def collision_detect(robot_id, delta_radian, line_speed):
    for i in range(4):
        if i != robot_id:
            d, _ = dist(map_info.robots[robot_id].loc, map_info.robots[i].loc)
            if d < 2.12 and \
                    abs(map_info.robots[robot_id].orientation) + abs(map_info.robots[i].orientation) > np.pi / 2:
                delta_radian = (delta_radian + (robot_id + 1) * np.pi / 4) % np.pi - np.pi
                break
    return delta_radian, line_speed


# 机器人的行为
def action():
    map_info.robots[0].forward(999)


if __name__ == '__main__':
    map_info = read_map()  # 读图
    # 这里使用finish()函数，把初始化信息写入到标准输出中
    finish()
    while True:
        # 读取一行信息
        line = sys.stdin.readline()
        # 如果读到空行，说明读完了，退出
        if not line:
            break
        parts = line.split(' ')  # 获取第一行信息
        frame_id = int(parts[0])  # 帧id
        map_info.money = int(parts[1])  # 当前资金
        read_util_ok()
        # 读取完毕，开始输出
        sys.stdout.write('%d\n' % frame_id)

        action()
        finish()

# 界面元素说明：
# …… ：表示 9 种工作台，右下角出现钻石图标 时，表示该工作台有产品可用于购买。
# 工作台上面的绿色进度条表示生产进度，底下的数字表示材料格状态，黑色表示该材料为空，红色表示该材料已有。
# ：表示机器人，机器人在携带物品时会变大，并且头上会出现一个数字和血条，
# 数字表示携带物品 ID，血条表示物品的价值系数比例，会随着时间和碰撞降低。

# 获取的每一帧信息：
# 1144 199346                       帧序号，当前金钱数
# 9                                 工作台数量

# 工作台类型，坐标x，坐标y，剩余生产时间，原材料格状态，产品格状态
# 1 43.75 49.25 0 0 1
# 2 45.75 49.25 0 0 1
# 3 47.75 49.25 0 0 1
# 4 43.75 47.25 -1 0 0
# 5 45.75 47.25 168 0 0
# 6 47.75 47.25 -1 0 0
# 7 44.75 45.25 -1 0 0
# 8 46.75 45.25 -1 0 0
# 9 46.25 42.25 -1 0 0

# 机器人所处工作台ID，携带物品类型，时间价值系数，碰撞价值系数，角速度，线速度x，线速度y，朝向，坐标x，坐标y
# 5 3 0.9657950401 1 0 0 0 -0.3755806088 47.5760498 47.40252686
# -1 0 0 0 0 0 0 -0.006108176429 43.75140762 48.23157501
# -1 0 0 0 0 0 0 0 3.25 2.25
# -1 0 0 0 0 0 0 0 45.75 1.75
# OK


# 操作的每一帧信息：
# id                             帧序号
# command id (arg1)                指令编号
# OK                               操作结束标志

# 1140
# forward 0 4.5
# rotate 0 3.14159
# forward 2 5
# forward 3 -5
# sell 1
# buy 1
# OK

# 大多数语言会默认对输出数据做缓冲，因此在输出完一帧控制数据时，你应该主动 flush 标
# 准输出以避免判题器读不到数据导致超时。此外，由于标准输出已用于比赛交互，因此不
# 要往标准输出打日志等其他内容，以免造成命令解析错误。平台上没有任何写文件权限，
# 故正式提交版本不要写日志文件，你可以使用 stderr 将日志输出到控制台以方便调试。

# 空闲机器人的操作：
# 1. 去工作台购买材料，前提是机器人手上没有物品且该工作台已有成品；
# 2. 去工作台出售材料，前提是机器人手上有物品且该工作台原材料格对应的材料状态为"0"；
# 3. 销毁机器人当前手中的物品。


# 利润计算
# 本题中总共有 7 种物品，分别编号为 1-7。其中，1-3 是原材料物品，由 1-3 号工作台
# 直接产生；4-7 是加工品，由 4-7 号工作台通过其他的物品加工而成。每一种物品都有
# 一个购买价和原始售出价，价格表如表 3-2 所示。

# 本题中工作台有 9 种，分别编号为 1-9。其中，1-3 为原材料工作台，它们负责生产 1-3
# 号原材料而不收购其他材料。4-7 是加工工作台，收购原料的同时也生产对应的成品。
# 8、9 为消耗工作台，只收购不生产，各工作台的收购原材料情况和生成物品情况如表
