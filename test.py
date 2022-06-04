import os
from matplotlib import pyplot as plt 
from process import *
import numpy as np
import glob

from BCModel.arguments import get_common_args
from BCModel.net import BCNet
import torch

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

def rename(path):
    fileList=os.listdir(path)
    for file in fileList:
        oldname = path + file
        newname = path + 'bag_' + file
        # newDir = path + '20220310_{}'.format(i)
        os.rename(oldname, newname)
  
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
    "data_5": {                         # 小转盘 
        "limit": [-920, -820, 0],
        "index": 5,
        "LCDirec": 'right',
        "testBag": 'bag_20220326_1'
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
    for dir in ['./data/data_0', './data/data_1', './data/data_2', './data/data_6']:
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


def getFeature(juncDir, pointNum, modelPath, args):
    """ 从 boundary 获取网络输入"""
    point = [-96.1358, -1184.32]
    end_point = [-196.56,-1186.13]
    cos = -1
    sin = 1.28155e-05
    boundary = np.loadtxt(juncDir, delimiter=",", dtype="double")
    rot_bound = rot(boundary[:, :2], point=point, sin=sin, cos=cos)
    re = Reduce(pointNum=pointNum)
    reduce_bound = re.getReducePoint(rot_bound)
    reduce_bound[:, 0] /= 10.
    
    bound = rot(tra=reduce_bound, point=[0, 0], sin=sin, cos=cos, rotDirec=1)
    # plt.plot(bound[:, 0], bound[:, 1])
    # c++最终结果
    # bound_as = np.loadtxt("./test/boundAS.csv", delimiter=",", dtype="double")
    # plt.plot(bound_as[:, 0], bound_as[:, 1], color='r')

    bound_cp = bsplineFitting(bound, cpNum=8, degree=3)
    # plt.scatter(bound_cp[:, 0], bound_cp[:, 1])
    print("cp: ", bound_cp)

    # bound_cp = np.array([[-0.027405,   -0.01127664],
    #                     [-0.26287144,  2.5372777 ],
    #                     [ 0.10681021,  3.6125014 ],
    #                     [ 1.4588654,   5.491921  ],
    #                     [ 2.2032552,   6.4668226 ],
    #                     [ 2.2953942,   7.640709  ],
    #                     [ 2.3508315,   8.499055  ],
    #                     [ 2.2691495,   9.146813  ],
    #                     [ 2.2605891,   9.39887   ]])
    # 加载模型
    model = BCNet(args.input_size, args.output_size, args)
    model.load_state_dict(torch.load(modelPath))
    print('load network successed')
    model.eval()

    feature = torch.FloatTensor(bound_cp).view(1, -1)
    pred = model(feature)
    pred = pred.view(-1, 2).detach().numpy()
    # plt.scatter(pred[:, 0], pred[:, 1])
    print("pred: ", pred)

    # 还原 控制点
    rot_cp = rot(tra=pred, point=pred[0, :], sin=sin, cos=cos, rotDirec=0)
    rot_cp[:, 0] *= 10.
    restore_cp = rot(tra=rot_cp, point=[0, 0], sin=sin, cos=cos, rotDirec=1)
    restore_cp[:, 0] += point[0]
    restore_cp[:, 1] += point[1]

    restore_cp[0, :] = point
    restore_cp[-1, :] = end_point
    print("restore_cp: ", restore_cp)
    plt.scatter(restore_cp[:, 0], restore_cp[:, 1])

    bs = BS_curve(n=8, p=3)        # 初始化B样条
    bs.get_knots()  
    x_asis = np.linspace(0, 1, 101)
    #设置控制点
    bs.cp = restore_cp       # 标签(控制点)
    res_curves = bs.bs(x_asis)
    plt.plot(res_curves[:, 0], res_curves[:, 1])

    plotMap(juncDir="./data/data_0/junction")

def showCPP():
    cp = np.loadtxt("./test/pred_cp.csv", delimiter=",", dtype="double")
    bs_curve = np.loadtxt("./test/bs_curve.csv", delimiter=",", dtype="double")
    plt.scatter(cp[:, 0], cp[:, 1])
    plt.plot(bs_curve[:, 0], bs_curve[:, 1])
    plotMap(juncDir="./data/data_0/junction")


if __name__ == '__main__':
    
    # run(isAug=True)

    # args = get_common_args()
    # modelPath = './model/2205_091659/episodes_999.pth'
    # getFeature("./test/tra.csv", pointNum=20, modelPath=modelPath, args=args)

    # showCPP()
    plotMap(juncDir="./data/data_6/junction")
#################################################################################
# 测试函数

    juncName = "data_5"
    bagName = config[juncName]['testBag']

    dataDir = './data/{}'.format(juncName)
    juncDir = './data/{}/junction'.format(juncName)
    traDir = './data/{}/{}'.format(juncName, bagName)
    index = config[juncName]['index']
    LCDirec = config[juncName]['LCDirec']

    # plotMap(juncDir=juncDir, traDir=traDir)

    # boundary = np.loadtxt("./data/test/boundary.csv", delimiter=",", dtype="double")
    # start_time = time.time()
    # bsplineFitting(tra=boundary, cpNum=8, degree=3, show=True)
    # end_time = time.time()
    # print("time 1: ", end_time - start_time)
    # start_time = time.time()
    # re = Reduce(pointNum=20)
    # reduce_bound = re.getReducePoint(boundary)
    # bsplineFitting(tra=reduce_bound, cpNum=8, degree=3, show=True)
    # end_time = time.time()
    # print("time 2: ", end_time - start_time)


    # plt.plot(boundary[:, 0], boundary[:, 1])
    # plt.show()
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