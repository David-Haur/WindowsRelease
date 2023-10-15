import sys
import numpy as np


class Map():
    def __init__(self):
        self.width = 50  # 地图长度
        self.height = 50  # 地图宽度
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

    # def reset(self):
    #    self.wt_to_be_purchased = []
    #    self.wt_to_be_added = []


class Robot():
    def __init__(self):
        self.id = -1  # id
        # self.radius = 0.45
        # self.radius_with_goods = 0.53
        # self.density = 20
        # self.quality = np.pi * np.square(self.radius) * self.density
        # self.quality_with_goods = np.pi * \
        #     np.square(self.radius_with_goods) * self.density
        self.on_task_dest = -1  # the worktable robot heading for 机器人的任务目标（以工作台的编号为目标），-1表示没有任务
        self.task = 0  # 1 buy, 2 sell, 3   机器人的任务：买、卖以及销毁携带的物品
        self.check_flag = False # 检查机器人是否携带有物品：False是没有，True是有

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
        self.pointx = 0 #机器人该去路径上的第几个点

    # The robot should go to the corresponding worktable(s) when it carries goods
    # 机器人在携带物品时，它能走向的对应工作台列表
    def get_optional_wt(self):
        ori_material = [0, 1, 2, 3, 4, 5, 6, 7]  # 能够携带的材料列表
        optional_worktables = [
            [], [4, 5, 9], [4, 6, 9], [5, 6, 9], [7, 9], [7, 9], [7, 9], [8, 9]
        ]  # 每种原材料分别有哪些工作台在使用（原材料序号从下标0开始），一共有7种原材料：例如1号原材料被4、5、9三个工作台需要
        self.optional_wt = optional_worktables[ori_material[self.goods_type]]

    # 机器人的行为：
    def forward(self, line_speed):
        sys.stdout.write('forward %d %f\n' % (self.id, line_speed))

    def rotate(self, angle_speed):
        sys.stdout.write('rotate %d %f\n' % (self.id, angle_speed))

    def buy(self):
        sys.stdout.write('buy %d\n' % self.id)

    def sell(self):
        sys.stdout.write('sell %d\n' % self.id)

    def destroy(self):
        sys.stdout.write('destroy %d\n' % self.id)


class Worktable():
    # 初始化工作台
    def __init__(self, type):
        self.type = type  # 工作台类型（1~9）
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
        self.get_info(type)  # 获得每个工作台的信息

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
def read_map():  # 初始化地图信息
    map_info = Map()
    row = 0
    worktable_num = 0
    robot_count = 0
    while True:
        col = 0
        line = input()
        if line == "OK":
            map_info.worktable_num = worktable_num
            return map_info
        for char in line:
            if char >= "1" and char <= "9":  # 工作台
                worktable = Worktable(int(char))
                worktable.id = worktable_num  # 根据遇到的顺序给编号
                worktable.loc = np.array([0.25 + col * 0.5, 49.75 - row * 0.5])  # 算工作台中心位置坐标
                map_info.worktables_list.append(worktable)  # 添加工作台实体
                worktable_num += 1
            elif char == "A":  # 机器人
                robot = Robot()
                robot.id = robot_count
                robot.loc = np.array([0.25 + col * 0.5, 49.75 - row * 0.5])
                map_info.robots.append(robot)
                robot_count += 1
            col += 1
        row += 1


def read_util_ok():  # 输入信息
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


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()


def dist(ori, dest):
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


# 机器人管理程序——让没有任务的机器人去有产品的工作台
def task_manager():
    for i in range(4):  # 遍历四个机器人
        if map_info.robots[i].on_task_dest == -1:  # 如果该机器人没有任务：
            if not map_info.robots[i].goods_type:  # 且机器人没有物品携带
                # 现在看哪个工作台有产品，那么就让机器人去哪个工作台，先进行4567的判断，可以使得利润更高。
                if len(map_info.wt_to_be_purchased_4567):  # 同时，如果有加工工作台已经生产出产品了，说明得给机器人一个任务，让它走到对应工作台的地方
                    # 现在开始在符合条件的工作台中找一个最近的出来：
                    min_dist_wt_4567 = float('inf')
                    for wt_4567 in map_info.wt_to_be_purchased_4567:
                        tmp_dist, _ = dist(map_info.robots[i].loc,
                                           map_info.worktables_list[wt_4567].loc)  # 求当前机器人的位置到某个工作台位置的距离和方位角
                        if tmp_dist < min_dist_wt_4567:
                            min_dist_wt_4567 = tmp_dist
                            min_index = wt_4567
                    # 找到了！然后把工作台实体提取出来，准备进行购买：
                    wt = map_info.worktables_list[min_index]  # 提取工作台实体
                    map_info.wt_to_be_purchased_4567.remove(
                        min_index)  # 机器人从这个工作台买走产品，那么wt_to_be_purchased_4567列表中就要删除对应工作台（先发生）
                    map_info.robots[i].on_task_dest = wt.id  # 给机器人任务：去那个最近的有产品的工作台（后发生）
                    map_info.robots[i].task = 1  # 更改机器人状态
                    wt.on_task_robot = i  # 更改那个最近工作台的状态
                elif len(map_info.wt_to_be_purchased_123):  # 同时，如果有原材料工作台已经生产出产品：
                    # 以下代码和上述意思一致！
                    min_dist_wt_123 = float('inf')
                    for wt_123 in map_info.wt_to_be_purchased_123:
                        tmp_dist, _ = dist(map_info.robots[i].loc, map_info.worktables_list[wt_123].loc)
                        if tmp_dist < min_dist_wt_123:
                            min_dist_wt_123 = tmp_dist
                            min_index = wt_123
                    wt = map_info.worktables_list[min_index]
                    map_info.wt_to_be_purchased_123.remove(min_index)
                    map_info.robots[i].on_task_dest = wt.id
                    map_info.robots[i].task = 1
                    wt.on_task_robot = i
                # 如果没有任何一个工作台有产品出来，不给任何机器人分配任务
            # 如果机器人有物品携带，说明它知道要去哪（走工作台的to_be_purchased方法），那我们不给它分配任务，跳过这个机器人
        # 如果机器人有任务，就跳过这个机器人，进行下一轮循环

        # else:
        #     map_info.robots[i].get_optional_wt()
        #     min_dist_wt = float('inf')
        #     for wt_id in map_info.wt_to_be_added:
        #         wt = map_info.worktables_list[wt_id]
        #         if wt.on_task_robot != -1:
        #             continue
        #         if wt.type in map_info.robots[i].optional_wt:
        #             if wt.raw_material_status[map_info.robots[i].goods_type] == '0':
        #                    tmp_dist, _ = dist(map_info.robots[i].loc, wt.loc)
        #                    if tmp_dist < min_dist_wt:
        #                        min_dist_wt = tmp_dist
        #                        min_wt = wt
        #     if min_dist_wt == float('inf'): # destroy
        #         map_info.robots[i].task = 3
        #     else:
        #         map_info.robots[i].on_task_dest = min_wt.id
        #         map_info.robots[i].task = 2
        #         min_wt.on_task_robot = i
        #         map_info.wt_to_be_added.remove(min_wt.id)


# 改变特定机器人的任务——如果是买入，把它的任务状态变成卖给下一个工作台，然后退出；
# 如果是卖出，那么重置机器人状态，变成初始状态。
def reset(robot_id):
    # 如果当前机器人的状态是购买，说明在某处工作台已经拿到产品了
    if map_info.robots[robot_id].task == 1:
        tmp_wt_id = map_info.robots[robot_id].on_task_dest  # 存一下机器人当前任务目标（工作台的编号）
        map_info.robots[robot_id].on_task_dest = \
            map_info.worktables_list[map_info.robots[robot_id].on_task_dest]\
                .id_sell_after_buy   # 让机器人的任务目标从当前工作台变成下一个工作台（必须满足下一个sell_after_buy工作台的原材料是当前工作台的产品）
        # 对原来工作台的处理：
        map_info.worktables_list[tmp_wt_id].id_sell_after_buy = -1  # 那么对于原先机器人的目标工作台，由于产品被买走了，它的下一个sell_after_buy工作台就变成-1
        map_info.worktables_list[tmp_wt_id].on_task_robot = -1  # 它也不再被机器人占用
        # 对下一个sell_after_buy工作台的处理：
        map_info.worktables_list[map_info.robots[robot_id].on_task_dest].on_task_robot = robot_id   # 工作台被机器人占用
        # 机器人信息更新：
        map_info.robots[robot_id].task = 2  # 机器人任务变成卖出
        map_info.robots[robot_id].goods_type_info = map_info.robots[robot_id].goods_type    # 记住当前买入产品的产品信息
        map_info.robots[robot_id].check_flag = False # 机器人不再有物品
        return

    #如果当前机器人的任务是卖出，说明在目标工作台机器人已经把物品卖掉了
    if map_info.robots[robot_id].task == 2:
        map_info.worktables_list[map_info.robots[robot_id].on_task_dest].on_task_robot = -1 # 此工作台不再被占用
        map_info.worktables_list[map_info.robots[robot_id].on_task_dest].material_ready[
            map_info.robots[robot_id].goods_type_info] = 0  # 把目标工作台的原材料状态变成就绪【问题：为什么变成0，而不是变成1】
        map_info.wt_to_be_added[map_info.robots[robot_id].goods_type_info].remove(
            map_info.robots[robot_id].on_task_dest)  # 已经把产品卖给对应工作台了，该工作台不能再接受同类型的产品了，因此wt_to_be_added表格要把该工作台删掉

    # 卖出收尾工作结束，或者已经销毁了物品，现在对机器人的状态进行重置：
    map_info.robots[robot_id].goods_type_info = 0   # 忘记刚刚给出的信息
    map_info.robots[robot_id].on_task_dest = -1 # 机器人任务目标重置
    map_info.robots[robot_id].task = 0  # 机器人任务类型重置
    map_info.robots[robot_id].check_flag = False    # 机器人不再携带物品


# 检查机器人携带物品和目标工作台的产品信息是否匹配，不匹配则销毁
def check(robot_id):
    # 如果当前机器人有任务目标（要去的工作台编号）
    if map_info.robots[robot_id].on_task_dest > -1: # 0~8
        # 如果当前机器人的任务是买入
        if map_info.robots[robot_id].task == 1:  # buy
            # 如果机器人当前携带的产品类型与下一个工作台需要的产品类型匹配
            if map_info.robots[robot_id].goods_type == map_info.worktables_list[
                map_info.robots[robot_id].on_task_dest].type:
                reset(robot_id) # 设置机器人的任务状态
                return True
            # 如果不匹配，销毁之
            else:
                map_info.robots[robot_id].task = 3
                return False
        # 如果当前机器人的任务是卖出
        elif map_info.robots[robot_id].task == 2:  # sell
            if map_info.worktables_list[map_info.robots[robot_id].on_task_dest].raw_material_status[
                map_info.robots[robot_id].goods_type] == '0':   # 就是机器人要去的工作台的特定原材料还是空的，那么就设置机器人信息，否则销毁产品
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


# 实际操控机器人的程序
def action():
    for i in range(4):
        if map_info.robots[i].task == 3:  # destroy 销毁物品，并重置机器人
            map_info.robots[i].destroy()
            reset(i)

        #如果当前机器人有目标工作台
        elif map_info.robots[i].on_task_dest != -1:
            # 首先算出到目标工作台的距离和方位角
            d, dest_radian = dist(
                map_info.robots[i].loc, map_info.worktables_list[map_info.robots[i].on_task_dest].loc)
            # 拿到当前机器人的方向
            ori_radian = map_info.robots[i].orientation
            # 算出机器人转到目标方向的弧度（重要变量）
            delta_radian = dest_radian - ori_radian
            # 如果转动角度大于π，选择一个小角度进行转动
            if abs(delta_radian) > np.pi:
                delta_radian = delta_radian + \
                               2 * np.pi if delta_radian < 0 else delta_radian - 2 * np.pi
            # 如果转动角度小于90度
            if abs(delta_radian) <= np.pi / 2:  # line speed can only be generated when the delta_angle <= 90
                # s = (-36) / np.pi * delta_radian + 6
                s = 6   # s是线速度，初始设为6
                if d < 1.5: # 如果到目标的距离小于1.5，则把线速度降为2
                    s = 2
            # 如果转动角度在90~180度之间，线速度设为3.5
            else:
                s = 3.5
            # 如果机器人当前所在的工作台id和目标工作台id吻合，调整速度，开始买卖
            if d < 0.4 and map_info.robots[i].worktableID == map_info.worktables_list[
                map_info.robots[i].on_task_dest].id:
                s = 0.8
                # on_task_dest = map_info.robots[i].on_task_dest
                if map_info.robots[i].check_flag:
                    if check(i):
                        continue
                if map_info.robots[i].task == 1:
                    map_info.robots[i].buy()
                    map_info.robots[i].check_flag = True
                elif map_info.robots[i].task == 2:
                    map_info.robots[i].sell()
                    map_info.robots[i].check_flag = True
                # elif map_info.robots[i].task == 0 and map_info.robots[i].goods_type == 0:  # buy after sell
                #     if map_info.worktables_list[on_task_dest].on_task_robot == -1:
                #         if map_info.worktables_list[on_task_dest].production_status == 1:
                #             if on_task_dest in map_info.wt_to_be_purchased_123:
                #                 map_info.wt_to_be_purchased_123.remove(on_task_dest)
                #             elif on_task_dest in map_info.wt_to_be_purchased_4567:
                #                 map_info.wt_to_be_purchased_4567.remove(on_task_dest)
                #             map_info.robots[i].on_task_dest = on_task_dest
                #             map_info.robots[i].task = 1
                #             map_info.worktables_list[on_task_dest].on_task_robot = i

            delta_radian, s = collision_detect(i, delta_radian, s)
            map_info.robots[i].rotate(delta_radian / 0.02)
            map_info.robots[i].forward(s)


# 机器人沿着路线走
def walk_through_path(route: list):
    # robot_num = 0
    for i in range(4):
        num = map_info.robots[i].pointx
        pointTo = route[num]   # 机器人要走第几个点的坐标信息

        if action2(i, pointTo) < 0.1 and num < len(route)-1:   # 如果离下个点距离小于0.1，且不是最后一个点，说明该去下一个点了
            map_info.robots[i].pointx += 1


# 让指定机器人走去指定的坐标，返回到指定坐标的距离
def action2(robot_num, dest: tuple)-> float:
    d, dest_radian = dist(map_info.robots[robot_num].loc, dest)  # 到下一个点的距离和方位角
    # 拿到当前机器人的方向
    ori_radian = map_info.robots[robot_num].orientation
    # 算出机器人转到目标方向的弧度（重要变量）
    delta_radian = dest_radian - ori_radian
    # 如果转动角度大于π，选择一个小角度进行转动
    if abs(delta_radian) > np.pi:
        delta_radian = delta_radian + 2 * np.pi if delta_radian < 0 else delta_radian - 2 * np.pi
    # 转动到合适角度（在一个合适的误差范围内）
    if abs(delta_radian) > 0.005:
        map_info.robots[robot_num].rotate(delta_radian / 0.02)
    # 开始往前走
    else:
        if d > 0.15:  # 以速度为2，那么必须减速的距离得大于0.15m（肯定有点误差，之后调参就行）
            map_info.robots[robot_num].forward(2)
        else:
            map_info.robots[robot_num].forward(0)
    return d


if __name__ == '__main__':
    map_info = read_map()  # 读图
    finish()
    while True:
        line = sys.stdin.readline()  # 读入每一帧的信息
        if not line:
            break
        parts = line.split(' ')  # 获取第一行信息
        frame_id = int(parts[0])  # 帧id
        map_info.money = int(parts[1])  # 当前资金
        read_util_ok()  # 读取剩余信息
        sys.stdout.write('%d\n' % frame_id)  # 写当前帧
        task_manager()
        # action()
        walk_through_path([(10, 20), (20, 20), (20, 40), (10, 40), (10, 20)])
        # action2((30, 20))
        finish()
