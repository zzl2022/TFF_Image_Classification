# -*- coding: utf-8 -*-
# @Software: PyCharm
# @Author : Alfred
# @Time : 2022-05-20 19:41


import tensorflow as tf
import tensorflow_federated as tff
from CreateTFFData import client_data
from Model import create_keras_model
import matplotlib.pyplot as plt
from args import args_parser
args = args_parser()

#%% 生成训练集和验证集

# 生成训练集和验证集，当让部分客户端参与训练时，可在 for 循环处进行修改。
print("Datasets Loading ...")
train_data = [client_data(i,
                          Batch_size = args.BatchSize,
                          Img_Size = args.ImgSize,
                          train_flag=True,
                          train_split = args.TrainSplit
                          )
              for i in range(args.NumClients)]
val_data = [client_data(i,
                        Batch_size = args.BatchSize,
                        Img_Size = args.ImgSize,
                        train_flag=False,
                        train_split = args.TrainSplit
                        )
            for i in range(args.NumClients)]
print("Loaded")

#%% tff 训练

input_spec = train_data[0].element_spec
ImgShape = args.ImgSize + (3,)

# 由 keras 模型构建 tff 模型
def model_fn():
  keras_model = create_keras_model(IMG_SHAPE = ImgShape, NUM_CLASSES = args.NumClass)
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  )

# 建立联邦学习进程 fed_avg 和验证
fed_avg = tff.learning.build_federated_averaging_process(
    model_fn,
    #客户端优化器，只针对客户端本地模型进行更新优化
    client_optimizer_fn=lambda:tf.keras.optimizers.Adam(),
    #服务器端优化器，只针对服务器端全局模型进行更新优化
    server_optimizer_fn=lambda:tf.keras.optimizers.Adam()
    )
evaluation = tff.learning.build_federated_evaluation(model_fn)

# 模型训练
state = fed_avg.initialize() # 初始化联邦学习进程
all_metrics = [] # 记录每一轮的acc和loss等metrics信息
for round_num in range(1, args.NUM_ROUNDS+1):
    print("Federated learning training ...")

    state, metrics = fed_avg.next(state, train_data)     # next表示一轮联邦平均
    all_metrics.append(metrics) # 记录每一轮的acc和loss等metrics信息，方便后面绘图展示结果

    test_metrics = evaluation(state.model, val_data)

    print('The', round_num, '-th round: ', '\n',
           'Training loss：', metrics['train']['loss'], 'Training accuracy：', metrics['train']['sparse_categorical_accuracy'])
    print(' Validation loss：', test_metrics['loss'], 'Validation accuracy：', test_metrics['sparse_categorical_accuracy'], '\n')

#%% 画图查看训练过程并保存参数

import pickle

acc = []
loss = []
for i in all_metrics:
    train = i['train']
    acc.append(train['sparse_categorical_accuracy'])
    loss.append(train['loss'])
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(len(acc)),acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.savefig('Training Accuracy and Loss.jpg')

# 保存训练中的metrics参数，以备后续分析
f= open('all_metrics.pkl', 'wb')
pickle.dump(all_metrics, f)
f.close()

#%% 保存训练好的全局模型

fed_avg_model = create_keras_model(IMG_SHAPE = ImgShape, NUM_CLASSES = args.NumClass)
state.model.assign_weights_to(fed_avg_model) #提取全局模型的参数
fed_avg_model.save('fed_avg_model.h5')  #保存模型

#%% 加载训练好的全局模型，并预测一张图像

from tensorflow import keras

img = keras.preprocessing.image.load_img(
    "COVID-19 Radiography Database/COVID/COVID (23).png", target_size=args.ImgSize
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
from keras.models import load_model
model = load_model('fed_avg_model.h5')
predictions = model.predict_classes(img_array)
# predictions = model.predict(img_array)
print(predictions)







