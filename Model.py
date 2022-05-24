# -*- coding: utf-8 -*-
# @Software: PyCharm
# @Author : Alfred
# @Time : 2022-05-21 10:57

import tensorflow as tf


# 建立模型，此分类器模型参考了tensorflow官网上的图像分类器案例
def create_keras_model(IMG_SHAPE, NUM_CLASSES):
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES,activation = "softmax")])
    return model


