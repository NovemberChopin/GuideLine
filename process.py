
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from process_data.uniformization import uniformization, reducePoint
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

        plt.plot(xpoint, ypoint, color='g', linestyle='--')   # 中心线
        plt.plot(l_b_x, l_b_y, color='y')
        plt.plot(r_b_x, r_b_y, color='y')
    boundary = np.load("{}/boundary.npy".format(juncDir))
    plt.plot(boundary[:, 0], boundary[:, 1], color='r')
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
        lineDir = "{}/segment_{}.csv".format(juncDir, index)
        segment = np.loadtxt(lineDir, delimiter=",", dtype="double")
        segment = calcuBoundary(segment)
        np.save("{}/segment_{}".format(juncDir, index), segment)

    # 2: 计算截取后的道路边界信息 -> boundary.npy
    seg_1 = np.load("{}/segment_0.npy".format(juncDir))
    seg_2 = np.load("{}/segment_2.npy".format(juncDir))
    boundary = np.vstack([seg_1, seg_2])
    boundary = boundary[(limit[0] < boundary[:, limit[2]]) & (boundary[:, limit[2]] < limit[1]), :]
    np.save("{}/centerLane.npy".format(juncDir), boundary[:, :2])
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
        tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
        start_x = tra[0, 0]
        start_y = tra[0, 1]
        tra[:, 0] -= start_x
        tra[:, 1] -= start_y
        plt.plot(tra[:, 0], tra[:, 1], color='r')
    else:       # 打印 dataDir 下所有轨迹
        fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
        for file in fileDirs:
            tra = np.loadtxt("{}/tra.csv".format(file), delimiter=",", dtype="double")
            plt.plot(tra[:, 0], tra[:, 1], color='r')
    
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
        # plt.plot(boundary[:, 0], boundary[:, 1], color='r')
    plt.show()


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


def bsplineFitting(tra, cpNum, degree, distance, show=False):
    """
    使用B样条拟合轨迹点
    cpNum: 控制点个数
    degree: 阶数
    distance: 轨迹点抽取距离
    return: 控制点
    """
    # 获取左边界线拟合参数并简化轨迹点
    traPoint = uniformization(tra, distance)

    bs = BS_curve(cpNum, degree)
    paras = bs.estimate_parameters(traPoint)
    knots = bs.get_knots()
    if bs.check():
        cp = bs.approximation(traPoint)
    x_ticks = np.linspace(0,1,101)
    curves = bs.bs(x_ticks)
    if show:
        plt.scatter(traPoint[:, 0], traPoint[:, 1])
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


def getTrainData(tra, boundary, LCDirec):
    """
    数据处理流程，输入为截取后的数据
    tra: 车辆轨迹 (N, 4)
    boundary: 路段边界轨迹 (N, 2)
    """
    # 获取监督数据（轨迹的B样条控制点）
    # temp_x = tra[0, 0]      # 记录轨迹起始点坐标(全局坐标)
    # temp_y = tra[0, 1]
    # tra[:, 0] -= tra[0, 0]  # 使用相对坐标
    # tra[:, 1] -= tra[0, 1]
    # end_x = tra[-1, 0]      # 轨迹结束相对坐标
    # end_y = tra[-1, 1]
    start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    traCP = bsplineFitting(tra=tra[:, 0:2], cpNum=8, degree=3, distance=5, show=False)
    # boundary[:, 0] -= temp_x
    # boundary[:, 1] -= temp_y
    # 拟合道路边界
    boundaryCP = bsplineFitting(boundary, cpNum=8, degree=3, distance=3, show=False)
    boundaryCP = np.array(boundaryCP).reshape(1, -1)

    # fectures = np.array([0, 0, start_speed, end_x, end_y]).reshape(1, -1)
    # 开始点、开始速度、结束点
    measure = np.array([tra[0, 0], tra[0, 1], start_speed, tra[-1, 0], tra[-1, 1]]).reshape(1, -1)
    if LCDirec == 'left':
        cmd = np.array([1, 0]).reshape(1, -1)
    else:   cmd = np.array([0, 1]).reshape(1, -1)

    features = np.hstack([boundaryCP, measure, cmd])
    labels = np.array(traCP).reshape(1, -1)
    return features, labels


def batchProcess(dataDir, juncDir, index, LCDirec):
    '''
    批量处理数据
    index: 路段数据编号
    LCDirec: 变道方向(left, right)
    '''
    if not os.path.exists("./data_input"):
        os.makedirs("./data_input")
    fea = []
    lab = []
    fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
    boundary = np.load("{}/boundary.npy".format(juncDir))
    for file in fileDirs:
        # tra = np.load("{}/tra.npy".format(file))
        tra, boundary = transfor(juncDir=juncDir, traDir=file)
        features, labels = getTrainData(tra=tra, boundary=boundary, LCDirec=LCDirec)
        fea.append(features)
        lab.append(labels)

    fea = np.array(fea).flatten().reshape(len(fileDirs) , -1)
    lab = np.array(lab).flatten().reshape(len(fileDirs) , -1)
    
    np.save("{}/features_{}".format("./data_input", index), fea)
    np.save("{}/labels_{}".format("./data_input", index), lab)
    return fea, lab


def rotationTra(tra, point, angle):
    """ 输入一条轨迹，返回按 point 旋转 angle 角度后的轨迹 """
    newTra = np.zeros_like(tra)
    x0, y0 = point[0], point[1]
    # 逆时针
    # newTra[:, 0] = (tra[:, 0]-x0)*np.cos(angle) - (tra[:, 1]-y0)*np.sin(angle)
    # newTra[:, 1] = (tra[:, 0]-x0)*np.sin(angle) + (tra[:, 1]-y0)*np.cos(angle)
    # 顺时针
    newTra[:, 0] = (tra[:, 0]-x0)*np.cos(angle) + (tra[:, 1]-y0)*np.sin(angle)
    newTra[:, 1] = (tra[:, 1]-y0)*np.cos(angle) - (tra[:, 0]-x0)*np.sin(angle)
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


def getAugmentTrainData(juncDir, traDir, step):
    """ 返回对一条数据旋转一周所得到的数据的网络输入 """
    features, labels = [], []
    dataNum = int(360 / step)
    for index in np.arange(start=0, stop=360, step=step):
        # 每旋转 5度 生成一条数据
        angle = np.pi * (index/180.)
        tra, boundary = augmentData(juncDir=juncDir, traDir=traDir, angle=angle)
        plt.plot(tra[:, 0], tra[:, 1])
        fea, lab = getTrainData(tra=tra, boundary=boundary)
        features.append(fea)
        labels.append(lab)
    plt.show()
    features = np.array(features).flatten().reshape(dataNum, -1)
    labels = np.array(labels).flatten().reshape(dataNum, -1)
    return features, labels


def batchAugProcess(dataDir, index, step):
    """
    处理 dataDir 下所有数据
    index: 保存训练数据的后缀
    step: 每隔 step 度生成一条数据
    """
    juncDir = "{}/junction".format(dataDir)
    fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
    features = np.zeros(shape=(1, 23))
    labels = np.zeros(shape=(1, 18))
    for file in fileDirs:
        fea, lab = getAugmentTrainData(juncDir=juncDir, traDir=file, step=step)
        print(file, ":", fea.shape, " ", lab.shape)
        features = np.vstack([features, fea])
        labels = np.vstack([labels, lab])
    features = np.delete(features, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    np.save("{}/features_aug_{}".format("./data_input", index), features)
    np.save("{}/labels_aug_{}".format("./data_input", index), labels)
    print("feas shape: ", features.shape, " labs shape: ", labels.shape)


def rot(tra, point, sin, cos, rotDirec):
    """ 
    顺时针旋转 
    rotDirec: 旋转方向。0: 顺时针。1: 逆时针
    """
    newTra = np.zeros_like(tra)
    x0, y0 = point[0], point[1]
    if rotDirec == 0:   # 顺时针
        newTra[:, 0] = (tra[:, 0]-x0)*cos + (tra[:, 1]-y0)*sin
        newTra[:, 1] = (tra[:, 1]-y0)*cos - (tra[:, 0]-x0)*sin
    if rotDirec == 1:   # 逆时针
        newTra[:, 0] = (tra[:, 0]-x0)*cos - (tra[:, 1]-y0)*sin
        newTra[:, 1] = (tra[:, 0]-x0)*sin + (tra[:, 1]-y0)*cos
    newTra[:, 2:4] = tra[:, 2:4]
    return newTra


def transfor(juncDir, traDir, show=False):
    """
    变换坐标使得车道中心线第一个点的朝 x 轴正向
    return: 变换后的轨迹tra和边界boundary
    """
    begin_seg = np.loadtxt("{}/segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    centerLane = np.load("{}/centerLane.npy".format(juncDir))
    point = [centerLane[0, 0], centerLane[0, 1]]
    cos = begin_seg[0, 2]
    sin = begin_seg[0, 3]

    boundary = np.load("{}/boundary.npy".format(juncDir))
    tra = np.load("{}/tra.npy".format(traDir))
    newTra = rot(tra, point=point, sin=sin, cos=cos, rotDirec=0)
    newBound = rot(boundary, point=point, sin=sin, cos=cos, rotDirec=0)
    if show:
        # 绘制旋转后的路段信息
        fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
        for file in fileDirs:
            lane = np.load(file)
            centerLine = rot(tra=lane[:, :2], point=point, sin=sin, cos=cos, rotDirec=0)
            leftLine = rot(tra=lane[:, 2:4], point=point, sin=sin, cos=cos, rotDirec=0)
            rightLine = rot(tra=lane[:, 4:6], point=point, sin=sin, cos=cos, rotDirec=0)
            newLane = np.hstack([centerLine, leftLine, rightLine])
            if show:
                plt.plot(newLane[:, 0], newLane[:, 1], color='g', linestyle='--')
                plt.plot(newLane[:, 2], newLane[:, 3], color='b')
                plt.plot(newLane[:, 4], newLane[:, 5], color='b')

        plt.plot(newTra[:, 0], newTra[:, 1], color='r')         # 新的轨迹
        plt.plot(newBound[:, 0], newBound[:, 1], color='r')     # 新边界
        pltTra(juncDir=juncDir, traDir=traDir)                  # 原有的路段信息
        plt.show()
    return newTra, newBound