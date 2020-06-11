import numpy as np
import pandas as pd

def main():
    #建立训练集、测试集

    #测试数据所在位置
    samp=pd.read_csv("D:\samp.txt")

    samp.head()

    samp_x=samp.iloc[:,[0,1,2,3]]
    samp_y=samp.iloc[:,[4]]

    np.random.seed(7)
    indices=np.random.permutation(len(samp_x))

    samp_x_train=samp_x.iloc[indices[0:130]]
    samp_y_train=samp_y.iloc[indices[0:130]]

    samp_x_test=samp_x.iloc[indices[130:150]]
    samp_y_test=samp_y.iloc[indices[130:150]]

 # 将dataframe格式的数据转换为numpy array格式，便于 调用函数计算
    samp_x_train=np.array(samp_x_train)
    samp_y_train=np.array(samp_y_train)

    samp_x_test=np.array(samp_x_test)
    samp_y_test=np.array(samp_y_test)

# 将labels的形状设置为(130,)
    samp_y_train.shape=(130,)
#将训练集、测试集带入自定义KNN分类器进行分类
    test_index = 12
    predict = KNNClassify(samp_x_test[test_index], samp_x_train, samp_y_train, 3, "distance")
    print(predict)
    print("新输入的实际类别是：", samp_y_test[test_index])
    print("\n")
    if predict == samp_y_test[test_index]:
        print("预测准确!")
    else:
        print("预测错误！")




# newInput: 新输入的待分类数据(x_test)，本分类器一次只能对一个新输入分类
# dataset：输入的训练数据集(x_train),array类型，每一行为一个输入训练集
# labels：输入训练集对应的类别标签(y_train)，格式为['A','B']而不是[['A'],['B']]
# k：近邻数
# weight：决策规则，"uniform" 多数表决法，"distance" 距离加权表决法

def KNNClassify(newInput, dataset, labels, k, weight):
    numSamples=dataset.shape[0]
    
    """step1: 计算待分类数据与训练集各数据点的距离（欧氏距离：距离差值平方和开根号）"""
    diff=np.tile(newInput,(numSamples,1)) - dataset
    squaredist=diff**2
    distance = (squaredist.sum(axis=1))**0.5
    
    """step2：将距离按升序排序，并取距离最近的k个近邻点"""
    # 对数组distance按升序排序，返回数组排序后的值对应的索引值
    sortedDistance=distance.argsort() 
    
    # 定义一个空字典，存放k个近邻点的分类计数
    classCount={}
    
    # 对k个近邻点分类计数，多数表决法
    for i in range(k):
        # 第i个近邻点在distance数组中的索引,对应的分类
        votelabel=labels[sortedDistance[i]]
        if weight=="uniform":
            # votelabel作为字典的key，对相同的key值累加（多数表决法）
            classCount[votelabel]=classCount.get(votelabel,0)+1 
        elif weight=="distance":
            # 对相同的key值按距离加权累加（加权表决法）
            classCount[votelabel]=classCount.get(votelabel,0)+(1/distance[sortedDistance[i]])
        else:
            print ("分类决策规则错误！")
            print ("\"uniform\"多数表决法\"distance\"距离加权表决法")
            break 
            
    # 对k个近邻点的分类计数按降序排序，返回得票数最多的分类结果
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    if weight=="uniform":
        print ("新输入到训练集的最近%d个点的计数为："%k,"\n",classCount)
        print ("新输入的类别是:", sortedClassCount[0][0])
    
    elif weight=="distance":
        print ("新输入到训练集的最近%d个点的距离加权计数为："%k,"\n",classCount)
        print ("新输入的类别是:", sortedClassCount[0][0])
    return sortedClassCount[0][0]



if __name__=='__main__':
    main()





    #原文https://blog.csdn.net/sudden2012/article/details/82810500
    #仅用作机器学习作业使用