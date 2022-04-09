import numpy as np

from BCModel.arguments import get_common_args
from BCModel.net import BCNet

import torch

from matplotlib import pyplot as plt 

from process_data.B_Spline_Approximation import  BS_curve
from process import *
from test import config


args = get_common_args()



def eval(feature, label, juncDir, traDir, modelPath, cpNum, degree):
    """
    查看模型预测效果
    juncDir: 车道轨迹路径
    traDir: 车辆轨迹路径
    cpNum, degree: B样条控制点与阶数
    distance: 抽稀距离
    """
    # 加载模型
    model = BCNet(args.input_size, args.output_size, args)
    model.load_state_dict(torch.load(modelPath))
    print('load network successed')
    model.eval()

    feature = torch.FloatTensor(feature[:, :23]).view(1, -1)
    pred = model(feature)

    # loss_function = nn.MSELoss(reduction='sum')
    # loss = loss_function(pred, torch.FloatTensor(label).view(1, -1))
    # print("loss is MSE: ", loss)

    # 将label  和pred  都转为numpy
    label = label.reshape(-1, 2)
    pred = pred.view(-1, 2).detach().numpy()
    print("pred :{}".format(pred))
    print("label : {}".format(label))
    loss = np.sum(pred-label)**2
    print("loss is: ", loss)
    
    bs = BS_curve(n=cpNum, p=degree)        # 初始化B样条
   
    # 拿到轨迹的开始位置
    tra = np.load("{}/tra.npy".format(traDir))
    tra = tra[:, 0:2]
    start_x, start_y = tra[0, 0], tra[0, 1] # 开始位置
    tra[:, 0] -= tra[0, 0]                  # 相对坐标
    tra[:, 1] -= tra[0, 1]
    # tra = uniformization(tra, distance)     # 抽稀
    bs.get_knots()                          # 计算b样条节点并设置
   
    x_asis = np.linspace(0, 1, 101)
    #设置控制点
    bs.cp = label       # 标签(控制点)
    curves_label = bs.bs(x_asis)
    curves_label[:, 0] += start_x   # 把数据恢复为地图位置
    curves_label[:, 1] += start_y

    bs.cp = pred        # 网络输出
    curves_pred = bs.bs(x_asis)
    curves_pred[:, 0] += start_x
    curves_pred[:, 1] += start_y
    # 保存预测的轨迹数据
    # np.save("{}/tra_pred".format(traDir), curves_pred)

    tra[:, 0] += start_x        # 把轨迹恢复为地图位置
    tra[:, 1] += start_y
    # plt.scatter(tra[:, 0], tra[:, 1])
    plt.plot(curves_pred[:, 0], curves_pred[:, 1], color='r')
    plt.plot(curves_label[:, 0], curves_label[:, 1], color='b')

    label[:, 0] += start_x
    label[:, 1] += start_y
    pred[:, 0] += start_x
    pred[:, 1] += start_y
    plt.scatter(label[:, 0], label[:, 1])
    plt.scatter(pred[:, 0], pred[:, 1], color='r')
    plotMap(juncDir=juncDir)    # 打印路段信息


def evalModel(modelPath):
    data_dirs=glob.glob(pathname='./data/*data*')
    print(data_dirs)
    for dir in ['./data/data_2', './data/data_6', './data/data_0']:
        sub_data = dir.split('/')[2]
        bagName = config[sub_data]['testBag']
        juncDir = '{}/junction'.format(dir)
        traDir = '{}/{}'.format(dir, bagName)

        tra, boundary = augmentData(juncDir=juncDir, traDir=traDir, angle=0)
        fea, lab = getTrainData(tra=tra, boundary=boundary)

        eval(
            feature=fea,
            label=lab,
            modelPath=modelPath,
            juncDir=juncDir, 
            traDir=traDir,
            cpNum=8, degree=3
        )

if __name__ == '__main__':

    modelPath = './model/2204_091042/episodes_999.pth'      # 所有路段截取固定长度100m
    evalModel(modelPath=modelPath)