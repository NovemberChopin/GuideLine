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
    fea_1 = np.load('./data_input/features_1.npy')
    fea_2 = np.load('./data_input/features_2.npy')
    features = np.vstack([fea_1, fea_2])
    print(features.shape)
    np.save('./data_input/features', features)

    lab_1 = np.load('./data_input/labels_1.npy')
    lab_2 = np.load('./data_input/labels_2.npy')
    labels = np.vstack([lab_1, lab_2])
    print(labels.shape)
    np.save('./data_input/labels', labels)

def fun3():
    features = np.load("./data_input/features.npy")
    labels = np.load("./data_input/labels.npy")
    cp = features[0, :18].reshape((9, -1))
    bs = BS_curve(8, 3)
    bs.get_knots()
   
    x_asis = np.linspace(0, 1, 101)
    
    bs.cp = cp
    curves = bs.bs(x_asis)
    plt.scatter(cp[:, 0], cp[:, 1])
    plt.plot(curves[:, 0], curves[:, 1])

    cp_lable = labels[0, :].reshape((9, -1))
    bs.cp = cp_lable
    curves_label = bs.bs(x_asis)
    plt.scatter(cp_lable[:, 0], cp_lable[:, 1])
    plt.plot(curves_label[:, 0], curves_label[:, 1])
    plt.show()




limitConfig = {
    "data_1": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1]     # y 轴坐标
}


if __name__ == '__main__':
    dataDir = './data'
    traDir = './data/bag_20220108_2'
    juncDir = './data/junction'
    limit = limitConfig['data_1']
    index = 1   # 区分生成的数据
    LCDirec = 'left'        # 左边换道

    # dataDir = './data2'
    # traDir = './data2/bag_20220127_4'
    # juncDir = './data2/junction'
    # limit = limitConfig['data_2']
    # index = 2               # 区分生成的数据
    # LCDirec = 'right'       # 右边换道
    # 打印路段信息
    # plotMap(traDir=traDir, juncDir=juncDir, segBegin=0, segEnd=0)

    # 路段数据预处理
    # preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)

    # 打印轨迹(相对坐标)
    # pltTra(dataDir=dataDir, juncDir=juncDir, traDir=traDir)     

    # ############
    # 处理一条数据
    # tra = np.load("{}/tra.npy".format(traDir))
    # boundary = np.load("{}/boundary.npy".format(juncDir))
    # fea, lab = getTrainData(tra=tra, boundary=boundary)
    # 对路段内所有数据进行处理
    # fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index, LCDirec=LCDirec)
    # print("fea shape: ", fea.shape, " lab shape: ", lab.shape)


    # ###########  旋转 增加数据 ##################
    # augmentData(juncDir=juncDir, traDir=traDir, angle=np.pi*(30/180), show=True)
    # feas, labs = getAugmentTrainData(juncDir=juncDir, traDir=traDir, step=5)
    # print(feas.shape, " ", labs.shape)
    # batchAugProcess(dataDir=dataDir, index=index, step=5)
    # ###########################################


    # fun3()
    fun()

    # transfor(juncDir=juncDir, traDir=traDir, show=True)