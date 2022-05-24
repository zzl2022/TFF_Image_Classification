# TFF_Image_Classification
X-ray image classification based on TensorFlow Federated for COVID-19 identification

基于 TensorFlow Federated 的 X 射线图像分类用于 COVID-19 识别

数据集：
COVID-19胸部X射线图像数据库包含有1200个COVID-19阳性图像，1341正常图像和1345病毒性肺炎图像。减压下载好的zip压缩包，将 "COVID-19 Radiography Database" 文件夹放在到项目文件夹下。数据集链接：https://www.heywhale.com/mw/dataset/6027caee891f960015c863d7

执行步骤：
首先，运行 DatasetPreprocess.py 文件，是将原始数据集分割为多个子数据集以模拟不同的客户端拥有不同的本地数据。此文件运行结束后产生新的数据集文件夹，以供接下来的联邦学习使用。
然后，运行 FederatedLlearning.py 文件，开始对图像分类任务进行联邦学习。此过程调用了 CreateTFFData.py 和 Model.py 文件。
项目涉及到的主要参数可在 args.py 文件中查看和修改。

环境：
Python 3.8、tensorflow-federated 0.17.0、tensorflow 2.3.4、keras 2.4.3

参考资料：
https://tensorflow.google.cn/federated/tutorials/federated_learning_for_image_classification
https://github.com/lpf111222/Federated-Learning-of-Image-classification-and-Its-simulation-in-TensorFlow-Federated
https://github.com/ki-ljl/FedAvg-numpy-pytorch-tff
