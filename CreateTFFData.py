# -*- coding: utf-8 -*-
# @Software: PyCharm
# @Author : Alfred
# @Time : 2022-05-20 15:36

import os
from PIL import Image
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from args import args_parser
args = args_parser()

def client_data(i, Batch_size, Img_Size, train_flag, train_split):
    """
    i: 客户机的 id
    Batch_size: 每个批次包括多少图像
    Img_Size: 将图像的尺寸统一
    train_flag: 当为 True 时生成训练集，当为 False 时生成验证集
    train_split: 训练集样本量在总样本中的占比
    """

    # 得到该客户端数据的路径
    subset_name = "Clients_data/Client_" + str(i)
    data_dir = os.path.join(os.getcwd(), subset_name) # data_dir 代表该客户端数据的路径

    # 统计该客户端的图像个数
    count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            count += 1

    # 读取数据
    data = [] # data 存放图像数据
    label = np.empty((count,), dtype="uint8") # label 存放图像标签
    i = 0 # 通过 i 控制图像标签在 label 中的位置
    j = -1 #  j 代表标签，由于以下 for 循环 从一个空的 root 开始，所以 j 初始值为 -1
    for root, dirs, files in os.walk(data_dir):

        # 获取一个类别文件夹下的图像名称
        ImageNames = [] # 一个类别文件夹下的全部图像名称放在 ImageNames 中
        ImageNames = ImageNames + files

        # 根据是训练集还是验证集对图像名称列表进行切片，ImageNamesSub 是子列表
        if train_flag:
            ImageNamesSub = ImageNames[:int(len(ImageNames)*train_split)]
        else:
            ImageNamesSub = ImageNames[int(len(ImageNames)*train_split):]

        # 逐张读取图像数据
        for ImageName in ImageNamesSub:
            img = image.load_img(os.path.join(root, ImageName))
            img = img.resize(Img_Size, Image.ANTIALIAS)
            img = image.img_to_array(img)
            data.append(img)
            label[i] = j
            i = i + 1
        j = j + 1

    # tf.data.Dataset.from_tensor_slices 函数的作用是接收tensor，
    # 对tensor的第一维度进行切分，并返回一个表示该tensor的切片数据集
    data = tf.data.Dataset.from_tensor_slices(data)
    label = tf.data.Dataset.from_tensor_slices(label)

    # 把 data 和 label 放在一起
    seq = tf.data.Dataset.zip((data, label))

    # 根据 Batch_size 生成最终的数据
    if train_flag:
        Epochs = args.Epochs    # 通过 Epochs 设置在每个客户端训练的周期
        seq = seq.batch(Batch_size, drop_remainder=True).shuffle(100).repeat(Epochs)
    else:
        seq = seq.batch(Batch_size, drop_remainder=True).shuffle(100)

    return seq



