#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import load_mnist


# データセットを読み込む
# 第一引数: 読み込むデータ数の割合, 第二引数: 使用する数字の下限, 第三引数: 使用する数字の上限
dataset = load_mnist.load_dataset(1, 0, 9)
# 訓練画像を取り出す
train_image = dataset['train_image']

# ランダムに訓練画像から20枚を選択
mask = np.random.choice(train_image.shape[0], 20)
select_image = train_image[mask]

# 20枚の画像を表示
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(select_image[i,:].reshape(28,28), cmap='gray')
plt.show()