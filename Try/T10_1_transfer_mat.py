import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# # 数据矩阵转图片的函数
# def MatrixToImage(data):
#     data = data*255
#     new_im = Image.fromarray(data.astype(np.uint8))
#     return new_im
#
# # 添加路径，metal文件夹下存放mental类的特征的多个.mat文件
# folder = '../ml/CrackForest-dataset-master/groundTruth/'
# path = os.listdir(folder)
# #print(os.listdir(r'/Users/hjy/Desktop/blues'))
#
# for each_mat in path:
#     if each_mat == '.DS_Store':
#         pass
#     else:
#         first_name, second_name = os.path.splitext(each_mat)
#         # 拆分.mat文件的前后缀名字，注意是**路径**
#         each_mat = os.path.join(folder, each_mat)
#         # print(each_mat)
#         array_struct = scio.loadmat(each_mat)
#         array_data = array_struct['groundTruth'] # 取出需要的数字矩阵部分
#         # df['count'].median().round().astype(int)
#         array_data = array_data *255
#         new_im = MatrixToImage(array_data)# 调用函数
#         plt.imshow(array_data, cmap=plt.cm.gray, interpolation='nearest')
#         new_im.show()
#         print(first_name)
#         new_im.save(first_name+'.bmp')# 保存图片

import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# 数据矩阵转图片的函数
def MatrixToImage(data):
    # new_im = Image.fromarray(data,mode='RGB')
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

folder = r'../ml/CrackForest-dataset-master/groundTruth/'
path = os.listdir(folder)
for each_mat in path:
    first_name, second_name = os.path.splitext(each_mat)
    # 拆分.mat文件的前后缀名字，注意是**路径**
    #yong yu fen ge wen jian min yu kuo zhan min
    each_mat = os.path.join(folder, each_mat)
    print(each_mat)
    array_struct = scio.loadmat(each_mat)
    fea_spa=array_struct['groundTruth'][0][0]
    for i in range(1 , len(fea_spa)):
        fea_spa_1=np.reshape(fea_spa[i],(320,480))
    #     fea_spa_1 = fea_spa
        print('i=',i)
        fea_spa_image = MatrixToImage(fea_spa_1)# 调用函数
        path = '../ml/CrackForest-dataset-master/mask/'  #+first_name+'/'
        if os.path.exists(path) is False:
            os.makedirs(path)
        fea_spa_image.save(path+first_name+'.jpg')# 保存图片

