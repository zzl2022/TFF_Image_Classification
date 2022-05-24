# -*- coding: utf-8 -*-
# @Software: PyCharm
# @Author : Alfred
# @Time : 2022-05-19 15:30

import os, shutil
from args import args_parser
args = args_parser()

subset_name = "Clients_data"

# 如果有存放客户端数据的文件夹则将其删除
for root, dirs, files in os.walk(os.getcwd()):
    if subset_name in dirs:
        shutil.rmtree(subset_name)

# 获得数据集的类别
data_dir = os.path.join(os.getcwd(), args.allset_name)
def file_name(file_dir):
  for root, dirs, files in os.walk(file_dir):
    a = dirs
    return a
a = file_name(data_dir)

# 分发数据给各个客户端
for i in range(0, args.NumClients):
    for category in a:

        # 统计原始文件夹下该类别下图像的名称和个数
        for root, dirs, files in os.walk(os.path.join(data_dir, category)):
            # 获取一个类别文件夹下的图像名称
            ImageNames = []  # 一个类别文件夹下的全部图像名称放在 ImageNames 中
            ImageNames = ImageNames + files
        count = len(ImageNames)

        # 新建在该客户端下存放该类别图像的文件夹
        dir = subset_name + "/Client_" + str(i) + "/" + category
        os.makedirs(dir)

        # 计算将哪些图像复制到新的文件夹下
        src_dir = os.path.join(data_dir, category)
        fnames = ImageNames[int(count*i/args.NumClients):int(count*(i+1)/args.NumClients)]

        # 将确定的图像复制到新的文件夹下，src 表示原始文件，dst 表示目标文件
        for fname in fnames:
            shutil.copyfile(src=src_dir + "/" + fname, dst=dir + "/" + fname)


