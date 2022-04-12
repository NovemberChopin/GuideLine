import os
from matplotlib import pyplot as plt 
from process import *
import numpy as np
import glob


def autoMkdir(first_dir, last_dir):
    """
    根据bag包文件名创建相应的文件夹
    first_dir: bag 包文件路径
    last_dir: 需要创建的文件夹路径
    """
    for root, dirs, files in os.walk(first_dir):
        for file in files:
            file = file.split('.b')[0]
            desDir = "{}/{}".format(last_dir, file)
            if not os.path.exists(desDir):
                os.mkdir(desDir)
    
def fun2(dataDir):
    """ 删除特定文件每条数据中的道路文件 """
    fileDirs = glob.glob(pathname = '{}/bag_2022*'.format(dataDir))
    for file in fileDirs:
        seg_file_list = glob.glob(pathname='{}/*.npy'.format(file))
        for seg_file in seg_file_list:
            try:
                os.remove(seg_file)
                print("delete file: ", seg_file)
            except:
                print("删除文件%s异常" % seg_file)

def fun():
    """
    把各个路段数据整合
    """
    fea_1 = np.load('./data_input/features_aug_1.npy')
    fea_2 = np.load('./data_input/features_aug_2.npy')
    fea_3 = np.load('./data_input/features_aug_3.npy')
    features = np.vstack([fea_1, fea_2, fea_3])
    print(features.shape)
    np.save('./data_input/features_aug', features)

    lab_1 = np.load('./data_input/labels_aug_1.npy')
    lab_2 = np.load('./data_input/labels_aug_2.npy')
    lab_3 = np.load('./data_input/labels_aug_3.npy')
    labels = np.vstack([lab_1, lab_2, lab_3])
    print(labels.shape)
    np.save('./data_input/labels_aug', labels)



config = {
    "data_0": {                         # 金蝶复兴四路
        "limit": [-200, -100, 0],       # x 轴坐标
        "index": 0,                     # 区分生成的数据
        "LCDirec": 'left',
        "testBag": 'bag_20220108_1'
    },
    "data_1": {                         # 十字路口 北
        "limit": [-3730, -3630, 1],
        "index": 1,
        "LCDirec": 'right',
        "testBag": 'bag_20220326_4'
    },
    "data_2": {                         # 十字路口 南
        "limit": [-3910, -3810, 1],
        "index": 2,
        "LCDirec": 'right',
        "testBag": 'bag_20220326_5'
    },
    "data_4": {                         # 十字路口 西
        "limit": [-590, -490, 0],
        "index": 4,
        "LCDirec": 'left',
        "testBag": 'bag_20220326_5'
    },
    "data_6": {                         # 最南端路口
        "limit": [-825, -725, 0],
        "index": 6,
        "LCDirec": 'left',
        "testBag": 'bag_20220108_1'
    },
}


def run(isAug=True):
    # 路段数据预处理
    features = np.zeros(shape=(1, 23))
    labels = np.zeros(shape=(1, 18))
    data_dirs=glob.glob(pathname='./data/*data*')
    print(data_dirs)
    for dir in ['./data/data_2', './data/data_6', './data/data_0']:
    # for dir in data_dirs:
        print(dir)
        sub_data = dir.split('/')[2]
        preProcess(dataDir=dir, 
                   limit=config[sub_data]['limit'], 
                   LCDirec=config[sub_data]['LCDirec'])

        # 对路段内所有数据进行处理
        # fea, lab = batchProcess(dataDir=dir, 
        #                         juncDir="{}/junction".format(dir), 
        #                         index=config[sub_data]['index'])
        # print("fea shape: ", fea.shape, " lab shape: ", lab.shape)

        # 扩充数据
        feas, labs = batchAugProcess(dataDir=dir, step=5, isAug=isAug)
        features = np.vstack([features, feas])
        labels = np.vstack([labels, labs])

    features = np.delete(features, 0, axis=0)
    labels = np.delete(labels, 0, axis=0)
    print("feas shape: ", features.shape, " labs shape: ", labels.shape)
    if isAug == True:
        np.save("{}/features_aug_nor".format("./data_input"), features)
        np.save("{}/labels_aug_nor".format("./data_input"), labels)


if __name__ == '__main__':

    run(isAug=True)
    
#################################################################################
# 测试函数

    juncName = "data_2"
    bagName = config[juncName]['testBag']

    dataDir = './data/{}'.format(juncName)
    juncDir = './data/{}/junction'.format(juncName)
    traDir = './data/{}/{}'.format(juncName, bagName)
    index = config[juncName]['index']
    LCDirec = config[juncName]['LCDirec']

    # plotMap(juncDir=juncDir, traDir=traDir)

    # 打印轨迹
    # pltTra(dataDir=dataDir, juncDir=juncDir, traDir=None)

    # 处理一条数据
    # tra = np.load("{}/tra.npy".format(traDir))
    # boundary = np.load("{}/boundary.npy".format(juncDir))
    # fea, lab = getTrainData(tra=tra, boundary=boundary)
    
    
    # augmentData(juncDir=juncDir, traDir=traDir, angle=np.pi*(30/180), show=True)
    # feas, labs = getAugData(juncDir=juncDir, traDir=traDir, dataNum=100)
    # feas, labs = getAugmentTrainData(juncDir=juncDir, traDir=traDir, step=5)
    # print(feas.shape, " ", labs.shape)

    # 转换航角
    # transfor(juncDir=juncDir, traDir=traDir, show=True)

    # centerLane = np.load("{}/centerLane.npy".format(juncDir))
    # point = [centerLane[0, 0], centerLane[0, 1]]
    # cos = centerLane[0, 2]
    # sin = centerLane[0, 3]
    # feas, labs = getAugmentTrainData(
    #         juncDir=juncDir, traDir=traDir, step=5, point=point, cos=cos, sin=sin)
    # print("fea shape: ", feas.shape, "lab shape: ", labs.shape)
    # print(feas[0, :])
    # print(feas[1, :])

    # lab = np.load("{}/label.npy".format(traDir))
    # lab = lab.reshape(-1, 2)
    # plt.plot(lab[:, 0], lab[:, 1])
    # plt.show()