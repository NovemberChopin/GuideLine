import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from process_data.uniformization import uniformization, Reduce
from process_data.B_Spline_Approximation import BS_curve
import math

def plotMap(juncDir, traDir=None, segBegin=0, segEnd=0, tra_begin=0, tra_length=0):
    """
    traDir: 轨迹路径
    juncDir: 道路节点数据路径
    tra_begin: 需要的打印轨迹的起始点
    tra_length: 需要打印的轨迹长度。0表示到结束
    """
    # 获取路径下文件夹下个数
    path_file_number=glob.glob(pathname='{}/*.csv'.format(juncDir))
    if segEnd == 0:
        segEnd = len(path_file_number)
    for index in range(segBegin, segEnd):
        filename = '{}/segment_{}.csv'.format(juncDir, index)
        data = np.loadtxt(filename, delimiter=",", dtype="double")
        xpoint = data[:,0]
        ypoint = data[:,1]
        cos = data[:, 2]
        sin = data[:, 3]
        lLength = data[:, 5]
        rLength = data[:, 7]
        # left boundary
        l_b_x = xpoint - lLength*sin
        l_b_y = ypoint + lLength*cos
        # right boundary
        r_b_x = xpoint + rLength*sin
        r_b_y = ypoint - rLength*cos
        if traDir:      # 如果轨迹路径不为空，则打印轨迹
            tra = np.load("{}/tra.npy".format(traDir))
            # tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
            if tra_length == 0:
                plt.plot(tra[tra_begin:, 0], tra[tra_begin:, 1], color='r')   # 轨迹
            else:
                tra_end = tra_begin + tra_length
                plt.plot(tra[tra_begin:tra_end, 0], tra[tra_begin:tra_end, 1], color='r')
        if index % 2 == 0:
            plt.plot(xpoint, ypoint, color='g', linestyle='--')   # 中心线
            plt.plot(l_b_x, l_b_y, color='y')
            plt.plot(r_b_x, r_b_y, color='y')
        else:
            plt.plot(xpoint, ypoint, color='b', linestyle='--')   # 中心线
            plt.plot(l_b_x, l_b_y, color='r')
            plt.plot(r_b_x, r_b_y, color='r')
    plt.show()


def preProcess(dataDir, limit, LCDirec):
    """
    dataDir: 路段数据根目录
    limit: 路段范围 limit[0]: 下界. limit[1]: 上界. limit[2]: 坐标轴
    LCDirec: lane change direction: 换道方向: left or right
    1: 计算junction路段的边界并保存为 segment_<>.npy 数据
    2: 计算截取后的道路边界信息 -> boundary.npy (N, 2)
    3: 获取dataDir下所有截取范围后的轨迹 tra.npy
    """
    juncDir = "{}/junction".format(dataDir)
    # 1: 计算junction路段的边界并保存为 segment_<>.npy 数据
    fileDirs = glob.glob(pathname='{}/segment*.csv'.format(juncDir))
    for index in range(len(fileDirs)):
        segDir = "{}/segment_{}.csv".format(juncDir, index)
        segment = np.loadtxt(segDir, delimiter=",", dtype="double")
        segment = calcuBoundary(segment)
        np.save("{}/segment_{}".format(juncDir, index), segment)
    # 保存第一个路段的中心线以及航角信息
    seg_0 = np.loadtxt("{}/segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    centerLane = seg_0[(limit[0] < seg_0[:, limit[2]]) & (seg_0[:, limit[2]] < limit[1]), :4]
    np.save("{}/centerLane.npy".format(juncDir), centerLane)

    # 2: 计算截取后的道路边界信息 -> boundary.npy
    seg_0 = np.load("{}/segment_0.npy".format(juncDir))
    seg_2 = np.load("{}/segment_2.npy".format(juncDir))
    boundary = np.vstack([seg_0, seg_2])
    boundary = boundary[(limit[0] < boundary[:, limit[2]]) & (boundary[:, limit[2]] < limit[1]), :]
    if LCDirec == 'left':   # 左边界
        np.save("{}/boundary.npy".format(juncDir), boundary[:, 2:4])
    else: np.save("{}/boundary.npy".format(juncDir), boundary[:, 4:6])

    # 3: 获取dataDir下所有截取范围后的轨迹 tra.npy
    traFileDirs = glob.glob(pathname='{}/bag_2022*_*'.format(dataDir))
    for traFile in traFileDirs:
        tra = np.loadtxt("{}/tra.csv".format(traFile), delimiter=",", dtype="double")
        tra = tra[(limit[0] < tra[:, limit[2]]) & (tra[:, limit[2]] < limit[1]), :]
        np.save("{}/tra.npy".format(traFile), tra)
        

def pltTra(juncDir, dataDir=None, traDir=None):
    """
    traDir==None: 打印 dataDir 下所有轨迹
    traDir!=None: 打印一条轨迹（相对坐标）
    """
    if traDir:  # 打印一条轨迹
        # tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
        tra = np.load("{}/tra.npy".format(traDir))
        start_x = tra[0, 0]
        start_y = tra[0, 1]
        tra[:, 0] -= start_x
        tra[:, 1] -= start_y
        plt.plot(tra[:, 0], tra[:, 1], color='r')
    else:       # 打印 dataDir 下所有轨迹
        fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
        for file in fileDirs:
            # tra = np.loadtxt("{}/tra.csv".format(file), delimiter=",", dtype="double")
            tra = np.load("{}/tra.npy".format(file))            
            plt.plot(tra[:, 0], tra[:, 1], color='c')
    
    fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
    for file in fileDirs:
        lane = np.load(file)
        if traDir:
            lane[:, [0, 2, 4]] -= start_x
            lane[:, [1, 3, 5]] -= start_y
        plt.plot(lane[:, 0], lane[:, 1], color='g', linestyle='--')
        plt.plot(lane[:, 2], lane[:, 3], color='b')
        plt.plot(lane[:, 4], lane[:, 5], color='b')
    if traDir:      # 绘制边界线
        boundary = np.load("{}/boundary.npy".format(juncDir))
        boundary[:, 0] -= start_x
        boundary[:, 1] -= start_y
        plt.plot(boundary[:, 0], boundary[:, 1], color='r')
    # plt.show()


def calcuBoundary(laneInfo):
    """
    输入一路段信息，计算边界轨迹。
    返回(中心线、左边界，右边界)数据 shape:(N, 6)
    """
    xpoint = laneInfo[:,0]
    ypoint = laneInfo[:,1]
    cos = laneInfo[:, 2]
    sin = laneInfo[:, 3]
    lLength = laneInfo[:, 5]
    rLength = laneInfo[:, 7]
    # left boundary
    l_b_x = xpoint - lLength*sin
    l_b_y = ypoint + lLength*cos
    # right boundary
    r_b_x = xpoint + rLength*sin
    r_b_y = ypoint - rLength*cos
    # laneInfo shape: (dataLength, 6) (中心线、左边界，右边界)
    return np.vstack([xpoint, ypoint, l_b_x, l_b_y, r_b_x, r_b_y]).T


def bsplineFitting(tra, cpNum, degree, show=False):
    """
    使用B样条拟合轨迹点
    cpNum: 控制点个数
    degree: 阶数
    distance: 轨迹点抽取距离
    return: 控制点
    """
    bs = BS_curve(cpNum, degree)
    paras = bs.estimate_parameters(tra)
    knots = bs.get_knots()
    if bs.check():
        cp = bs.approximation(tra)
    x_ticks = np.linspace(0,1,101)
    curves = bs.bs(x_ticks)
    if show:
        plt.scatter(tra[:, 0], tra[:, 1])
        plt.plot(curves[:, 0], curves[:, 1], color='r')
        plt.plot(cp[:, 0], cp[:, 1], color='y')
        plt.scatter(cp[:, 0], cp[:, 1], color='y')
        plt.show()
    return cp


def polyFitting(laneInfo):
    """
    使用多项式拟合轨迹
    degree: 多项式阶数
    """
    # 获取左边界线拟合参数
    boundary = uniformization(laneInfo[:, 2:4], 5)
    param = np.polyfit(boundary[:, 0], boundary[:, 1], 3)
    plt.scatter(boundary[:, 0], boundary[:, 1])
    x = boundary[:, 0]
    plt.plot(x, param[0]*x**3 + param[1]*x**2 + param[2]*x**1 + param[3], 'k--')
    plt.show()
    return param


def showOneTra(traDir):
    """ 打印一条轨迹 """
    tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
    point = tra[:, :2]
  
    point = uniformization(point, len=5)
    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])
    plt.plot(point[0,:],point[1, :], color='r')
    plt.show()


def rotationTra(tra, point, angle):
    """ 输入一条轨迹，返回按 point 逆时针旋转 angle 角度后的轨迹 """
    newTra = np.zeros_like(tra)
    x0, y0 = point[0], point[1]
    newTra[:, 0] = (tra[:, 0]-x0)*np.cos(angle) - (tra[:, 1]-y0)*np.sin(angle)
    newTra[:, 1] = (tra[:, 0]-x0)*np.sin(angle) + (tra[:, 1]-y0)*np.cos(angle)
    return newTra


def augmentData(juncDir, traDir, angle, show=False):
    """
    通过对原始数据根据轨迹起始点为原点旋转不同角度增加数据并将其返回
    traDir: 需要增强的数据
    juncDir: 路段信息路径
    return: 旋转后的轨迹tra和道路边界boundary
    """
    tra = np.load("{}/tra.npy".format(traDir))
    newTra = rotationTra(tra, point=tra[0, :2], angle=angle)
    
    # 对 boundary 数据进行旋转
    boundary = np.load("{}/boundary.npy".format(juncDir))
    NewBoundary = rotationTra(tra=boundary, point=tra[0, :2], angle=angle)
    
    if show:
        # 绘制旋转后的路段信息
        fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
        for file in fileDirs:
            lane = np.load(file)
            centerLine = rotationTra(tra=lane[:, :2], point=tra[0, :2], angle=angle)
            leftLine = rotationTra(tra=lane[:, 2:4], point=tra[0, :2], angle=angle)
            rightLine = rotationTra(tra=lane[:, 4:6], point=tra[0, :2], angle=angle)
            newLane = np.hstack([centerLine, leftLine, rightLine])
            if show:
                plt.plot(newLane[:, 0], newLane[:, 1], color='g', linestyle='--')
                plt.plot(newLane[:, 2], newLane[:, 3], color='b')
                plt.plot(newLane[:, 4], newLane[:, 5], color='b')

        plt.plot(newTra[:, 0], newTra[:, 1], color='r')             # 新轨迹
        plt.plot(NewBoundary[:, 0], NewBoundary[:, 1], color='r')   # 新边界
        pltTra(juncDir=juncDir, traDir=traDir)                      # 原有的路段信息
        plt.show()
    return newTra, NewBoundary


def buildTrainData(reduce_tra, reduce_bound, cos, sin, start_speed, rotDirec):
    """ 对抽稀后的轨迹逆时针旋转 angle 角度，然后求其控制点 """
    rot_tra = rot(tra=reduce_tra, point=[0, 0], cos=cos, sin=sin, rotDirec=rotDirec)
    lab_cp = bsplineFitting(rot_tra, cpNum=8, degree=3)     # shape: (9, 2)
    # plt.plot(rot_tra[:, 0], rot_tra[:, 1])
    rot_bound = rot(tra=reduce_bound, point=[0, 0], cos=cos, sin=sin, rotDirec=rotDirec)
    bound_cp = bsplineFitting(rot_bound, cpNum=8, degree=3)

    lab = lab_cp.reshape(1, -1)
    bound_cp = bound_cp.reshape(1, -1)
    measure = np.array([rot_tra[1, 0], rot_tra[1, 1], 
                        start_speed, rot_tra[-1, 0], rot_tra[-1, 1]]).reshape(1, -1)
    fea = np.hstack([measure, bound_cp])
    return fea, lab


def getAugmentTrainData(juncDir, traDir, step, point, cos, sin, isAug, pointNum=20):
    """ 返回对一条数据旋转一周所得到的数据的网络输入 """
    features, labels = [], []
    dataNum = int(360 / step) + 50
    # 旋转轨迹使得航角为0
    tra = np.load("{}/tra.npy".format(traDir))
    start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    newTra = rot(tra[:, :2], point=point, sin=sin, cos=cos)
    boundary = np.load("{}/boundary.npy".format(juncDir))
    newBoundary = rot(boundary, point=point, sin=sin, cos=cos)
    # TODO 数据抽稀
    re = Reduce(pointNum=pointNum)
    reduce_tra = re.getReducePoint(newTra)
    reduce_bound = re.getReducePoint(newBoundary)
    # TODO 把 x 轴缩放
    reduce_tra[:, 0] /= 10.
    reduce_bound[:, 0] /= 10.

    # 计算源数据的 fea 和 lab并保存，用作效果评估
    fea, lab = buildTrainData(
            reduce_tra=reduce_tra, reduce_bound=reduce_bound, cos=cos,
            sin=sin, start_speed=start_speed, rotDirec=1)
    np.save("{}/feature.npy".format(traDir), fea)
    np.save("{}/label.npy".format(traDir), lab)

    if isAug == True:
        # 每隔 step 度扩充数据
        for index in np.arange(start=0, stop=360, step=step):
            # TODO 旋转扩充数据
            angle = np.pi * (index/180.)
            rot_cos = np.cos(angle)
            rot_sin = np.sin(angle)
            fea, lab = buildTrainData(
                reduce_tra=reduce_tra, reduce_bound=reduce_bound, cos=rot_cos,
                sin=rot_sin, start_speed=start_speed, rotDirec=0)
            features.append(fea)
            labels.append(lab)
        # 再按随机角度生成 50 条数据
        angles = np.random.randint(low=1, high=360, size=50)
        for angle in angles:
            angle = np.pi * (index/180.)
            rot_cos = np.cos(angle)
            rot_sin = np.sin(angle)
            fea, lab = buildTrainData(
                reduce_tra=reduce_tra, reduce_bound=reduce_bound, cos=rot_cos,
                sin=rot_sin, start_speed=start_speed, rotDirec=0)
            features.append(fea)
            labels.append(lab)
        # plt.show()
        features = np.array(features).flatten().reshape(dataNum, -1)
        labels = np.array(labels).flatten().reshape(dataNum, -1)
    return features, labels


def batchAugProcess(dataDir, step, isAug):
    """
    处理 dataDir 下所有数据
    step: 每隔 step 度生成一条数据
    dataNum: 需要扩充的数据数量
    """
    # 对于每一个junction边界
    juncDir = "{}/junction".format(dataDir)
    fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
    features = np.zeros(shape=(1, 23))
    labels = np.zeros(shape=(1, 18))
    # TODO 航角归零: 对一个路段来说旋转点和角度是相同的
    centerLane = np.load("{}/centerLane.npy".format(juncDir))
    point = [centerLane[0, 0], centerLane[0, 1]]
    cos = centerLane[0, 2]
    sin = centerLane[0, 3]
    for file in fileDirs:
        fea, lab = getAugmentTrainData(
            juncDir=juncDir, traDir=file, step=step, point=point, cos=cos, sin=sin, isAug=isAug)
        if isAug:
            print(file, ":", fea.shape, " ", lab.shape)
            features = np.vstack([features, fea])
            labels = np.vstack([labels, lab])
    features = np.delete(features, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    print("data Dir: ", dataDir, "feas shape: ", features.shape, " labs shape: ", labels.shape)
    return features, labels


def rot(tra, point, sin, cos, rotDirec=0):
    """ 
    顺时针旋转 
    rotDirec: 旋转方向。0: 顺时针(默认)。1: 逆时针
    """
    newTra = np.zeros_like(tra)
    x0, y0 = point[0], point[1]
    if rotDirec == 0:   # 顺时针
        newTra[:, 0] = (tra[:, 0]-x0)*cos + (tra[:, 1]-y0)*sin
        newTra[:, 1] = (tra[:, 1]-y0)*cos - (tra[:, 0]-x0)*sin
    if rotDirec == 1:   # 逆时针
        newTra[:, 0] = (tra[:, 0]-x0)*cos - (tra[:, 1]-y0)*sin
        newTra[:, 1] = (tra[:, 0]-x0)*sin + (tra[:, 1]-y0)*cos
    return newTra


def transfor(juncDir, traDir, show=False):
    """
    变换坐标使得车道中心线第一个点的朝 x 轴正向
    return: 变换后的轨迹tra和边界boundary
    """
    centerLane = np.load("{}/centerLane.npy".format(juncDir))
    point = [centerLane[0, 0], centerLane[0, 1]]
    cos = centerLane[0, 2]
    sin = centerLane[0, 3]

    boundary = np.load("{}/boundary.npy".format(juncDir))
    tra = np.load("{}/tra.npy".format(traDir))
    newTra = rot(tra, point=point, sin=sin, cos=cos, rotDirec=0)
    newTra[:, 2:4] = tra[:, 2:4]
    newBound = rot(boundary, point=point, sin=sin, cos=cos, rotDirec=0)

    newTra[:, 0] /= 10.
    newBound[:, 0] /= 10.

    if show:
        # 绘制旋转后的路段信息
        fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
        for file in fileDirs:
            lane = np.load(file)
            centerLine = rot(tra=lane[:, :2], point=point, sin=sin, cos=cos, rotDirec=0)
            leftLine = rot(tra=lane[:, 2:4], point=point, sin=sin, cos=cos, rotDirec=0)
            rightLine = rot(tra=lane[:, 4:6], point=point, sin=sin, cos=cos, rotDirec=0)
            newLane = np.hstack([centerLine, leftLine, rightLine])
            newLane[:, [0, 2, 4]] /= 10.
            if show:
                plt.plot(newLane[:, 0], newLane[:, 1], color='g', linestyle='--')
                plt.plot(newLane[:, 2], newLane[:, 3], color='b')
                plt.plot(newLane[:, 4], newLane[:, 5], color='b')

        plt.plot(newTra[:, 0], newTra[:, 1], color='r')         # 新的轨迹
        plt.plot(newBound[:, 0], newBound[:, 1], color='r')     # 新边界
        pltTra(juncDir=juncDir, traDir=traDir)                  # 原有的路段信息
        plt.show()
    return newTra, newBound
    

