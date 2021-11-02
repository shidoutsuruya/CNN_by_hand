import os
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
cifar10_dir_path=r"D:\Python_data\cifar10\cifar-10-batches-py"
data_list=os.listdir(cifar10_dir_path)
data_1_path=os.path.join(cifar10_dir_path,data_list[1])
onehot=OneHotEncoder(sparse=False)
def load_x_and_y(data_path=data_1_path):
    """
    load the data from path
    """
    with open(data_path, 'rb') as fo:
        data = pickle.load(fo,encoding='bytes')#keys [b'batch_label', b'labels', b'data', b'filenames']
    #process x
    x=data[b'data'].reshape(-1,3,32,32)
    x=x.transpose(0,2,3,1)#3,32,32-->32,32,3
    #process y
    y_hot=onehot.fit_transform(np.array(data[b'labels']).reshape(-1,1))
    return x,y_hot
X,Y=load_x_and_y()
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=50,stratify=Y)
def accurate_curve(history):
    fig=plt.figure('hello')
    #ax1
    ax1=fig.add_subplot(2,1,1)
    ax1.set_title('accuracy')
    ax1.set_ylim(0,1)
    for i in ['accuracy','val_accuracy']:
        ax1.plot(history.epoch,history.history[i],label=i)
    ax1.grid()
    ax1.legend()
    #ax2
    ax2=fig.add_subplot(2,1,2)
    ax2.set_title('loss')
    for i in ['loss','val_loss']:
        ax2.plot(history.epoch,history.history[i],label=i)
    ax2.grid()
    ax2.legend()
    plt.show()
    