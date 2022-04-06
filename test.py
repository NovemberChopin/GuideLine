import os
from matplotlib import pyplot as plt 
from process import *
import numpy as np
import glob
from process_data.B_Spline_Approximation import BS_curve
from process_data.uniformization import uniformization






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
        seg_file_list = glob.glob(pathname='{}/segment*.csv'.format(file))
        for seg_file in seg_file_list:
            try:
                os.remove(seg_file)
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





limitConfig = {
    "data_1": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1],    # y 轴坐标
    "data_3": [-850, -700, 0]       # x 轴坐标
}


if __name__ == '__main__':
    dataDir = './data'
    traDir = './data/bag_20220326_5'
    juncDir = './data/junction'
    limit = limitConfig['data_1']
    index = 1               # 区分生成的数据
    LCDirec = 'left'        # 左边换道

    # dataDir = './data2'
    # traDir = './data2/bag_20220326_3'
    # juncDir = './data2/junction'
    # limit = limitConfig['data_2']
    # index = 2               # 区分生成的数据
    # LCDirec = 'right'       # 右边换道

    # dataDir = './data3'
    # traDir = './data3/bag_20220108_1'
    # juncDir = './data3/junction'
    # limit = limitConfig['data_3']
    # index = 3               # 区分生成的数据
    # LCDirec = 'left'        # 右边换道

    # 打印路段信息
    # plotMap(juncDir=juncDir, traDir=traDir, segBegin=0, segEnd=0)

    # 路段数据预处理
    # preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)

    # 打印轨迹(相对坐标)
    pltTra(dataDir=dataDir, juncDir=juncDir, traDir=None)

    # ############
    # 处理一条数据
    # tra = np.load("{}/tra.npy".format(traDir))
    # boundary = np.load("{}/boundary.npy".format(juncDir))
    # fea, lab = getTrainData(tra=tra, boundary=boundary)
    # 对路段内所有数据进行处理
    # fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index)
    # print("fea shape: ", fea.shape, " lab shape: ", lab.shape)


    # augmentData(juncDir=juncDir, traDir=traDir, angle=np.pi*(30/180), show=True)

    # feas, labs = getAugmentTrainData(juncDir=juncDir, traDir=traDir, step=5)
    # print(feas.shape, " ", labs.shape)

    # batchAugProcess(dataDir=dataDir, index=index, step=5)

    # fun()