
import math
import numpy as np
import matplotlib.pyplot as plt

class Reduce(object):
    def __init__(self, pointNum, low=1., high=10.):
        self.pointNum = pointNum
        self.low = low
        self.high = high
        self.dis = None
        self.saveList = None

    def getReducePoint(self, tra):
        l, r = self.low, self.high
        while l < r:
            mid = l + (r-l)/2.
            saveList = self.uniformization(tra=tra, dis=mid)
            if saveList.shape[0] == self.pointNum:
                self.dis = mid
                self.saveList = saveList
                break;  
            elif saveList.shape[0] > self.pointNum:
                # 间隔太近了，需要增加间隔
                l = mid
            else: r = mid
        return self.saveList
    
    def uniformization(self, tra, dis, show=False):
        """
        把密度不均匀的轨迹点均匀化（两点之间距离相近）
        tra: 轨迹点. shape: (N, 2|4)
        dis: 相两点之间的距离
        """
        n = tra.shape[0]
        i = 0
        saveList = []
        saveList.append(tra[i, :])  # 添加第一个点
        for j in range(1, n):
            # (x1-x2)**2 + (y1-y2)**2
            interval = math.sqrt((tra[i, 0]-tra[j, 0])**2 + (tra[i, 1]-tra[j, 1])**2)
            if interval > dis:
                saveList.append(tra[j, :])
                i = j
        saveList = np.array(saveList)
        if show:
            plt.scatter(saveList[:, 0], saveList[:, 1])
            plt.show()
        return saveList

def uniformization(tra, len, show=False):
    """
    把密度不均匀的轨迹点均匀化（两点之间距离相近）
    tra: 轨迹点. shape: (N, 2|4)
    len: 相两点之间的距离
    """
    # tra = np.loadtxt("{}tra.csv".format(traDir), delimiter=",", dtype="double")
    n = tra.shape[0]
    i = 0
    saveList = []
    saveList.append(tra[i, :])  # 添加第一个点
    for j in range(1, n):
        # (x1-x2)**2 + (y1-y2)**2
        dis = math.sqrt((tra[i, 0]-tra[j, 0])**2 + (tra[i, 1]-tra[j, 1])**2)
    
        if dis > len:
            saveList.append(tra[j, :])
            # print("dis = {}".format(dis))
            i = j
    saveList = np.array(saveList)
    # print("saveList len: ", saveList.shape)
    if show:
        plt.scatter(saveList[:, 0], saveList[:, 1])
        plt.show()
    return saveList




# 轨迹点抽稀的方法，弃用
def reducePoint(tra, step):
    """
    对轨迹点精简处理
    tra: 轨迹点 numpy (Num, 2)
    step: 缩小倍数
    """
    length = tra.shape[0]
    res = []
    for i in range(0, length, step):
        res.append(tra[i, :])
    return np.array(res)