import numpy as np
import pandas as pd
import matplotlib.pylab as plt
data_=pd.read_csv('vehicle.csv')
print(data_)
feature=np.array(data_.iloc[:,0:2])  # 将参数与特征进行分离，返回数据类型为数组,这里只拿去前两列
#print(feature)
labels = data_['label'].tolist()   # 将'label'标签提取出来转换为列表类型,方便后续使用
#print(labels)

# 数据可视化
plt.scatter(data_['length'][data_['label']=='car'],data_['width'][data_['label']=='car'],c='y')  #先取length的数值,里面有car和truck的长度,再单独取label那一行为car的值
plt.scatter(data_['length'][data_['label']=='truck'],data_['width'][data_['label']=='truck'],c='r')  #先取width的数值,里面有car和truck的长度,再单独取label那一行为truck的值
#print(data_['length'])
#print(plt.show())

test = [4.7,2.1] # 待测样本

numSamples = data_.shape[0]  # 读取矩阵的长度,这里是读取第一维的长度# 运行结果：150
diff_= np.tile(test,(numSamples,1)) #这里表示test列表竖向重复150次，横向重复1一次，组成一个素组
# numpy.tile(A,B)函数：A=[4.7,2.1]，B=（3,4）,意思是列表A在行方向（从上到下）重复3次，在列放心（从左到右）重复4次
diff = diff_-feature  # 利用这里的实验值和样本空间里的每一组数据进行相减
squreDiff = diff**2   # 将差值进行平方
squreDist = np.sum(squreDiff,axis=1)  # 每一列横向求和，最后返回一维数组
distance = squreDist ** 0.5 
sorteDisIndices = np.argsort(distance)  # 排序
k=9  # k个最近邻
classCount = {}  # 字典或者列表的数据类型需要声明
label_count=[]   # 字典或者列表的数据类型需要声明
for i in range(k):
    voteLabel = labels[sorteDisIndices[i]]
    classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    label_count.append(voteLabel)

from collections import Counter
word_counts = Counter(label_count)
top = word_counts.most_common(1)   # 返回数量最多的值以及对应的标签
#print(word_counts)
#print(top)


# 利用sklearn进行实现
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

data_=pd.read_csv('vehicle.csv')
feature=np.array(data_.iloc[:,0:2])  # 将参数与特征进行分离，返回数据类型为数组,这里只拿去前两列
labels = data_['label'].tolist()   # 将'label'标签提取出来转换为列表类型,方便后续使用

from sklearn.model_selection import train_test_split
feautre_train,feautre_test,label_train,label_test=train_test_split(feature,labels,test_size=0.2)
# 指出训练集的标签和特征以及测试集的标签和特征，0.2为参数，对测试集以及训练集按照2:8进行划分
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors= 9)

model.fit(feautre_train,label_train) # 现在只需要传入训练集的数据
prediction=model.predict(feautre_test)
#print(prediction)

labels=['car','truck']
classes=['car',
         'truck']
from sklearn.metrics import classification_report
result_=classification_report(label_test,prediction,target_names = classes,labels=labels,digits=4)
# target_names：类别；digits：int，输出浮点值的位数
print(result_)
