#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This program is based on the following program.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/trainer.py
# Copyright (c) 2016 Koki Saitoh
# The original program is released under the MIT License.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/LICENSE.md

# 必要なライブラリをインポート
import pickle
import numpy as np
import matplotlib.pyplot as plt
import load_mnist
from network import TwoLayerNetwork


print("データセットの読み込みを開始します.")
# データセットに使用する数字の上限を定義
upper = 2
# データセットに使用する数字の上限を定義
lower = 0
# データセットを読み込む
# 第一引数: 読み込むデータ数の割合, 第二引数: 使用する数字の下限, 第三引数: 使用する数字の上限
dataset = load_mnist.load_dataset(0.1, lower, upper)
# 訓練画像を取り出す
train_image = dataset['train_image']
# 訓練画像のラベルを取り出す
train_label = dataset['train_label']
# テスト画像を取り出す
test_image = dataset['test_image']
# テスト画像のラベルを取り出す
test_label = dataset['test_label']
print("訓練データ数: {0}, テストデータ数: {1}".format(train_image.shape[0], test_image.shape[0]))
print("データセット内にある数字: {0}".format(list(range(lower, upper + 1))))

# エポック数を定義
epoch = 10
# 訓練画像の数を取得
train_size = train_image.shape[0]
# バッチサイズを定義
batch_size = 50
# 学習率を定義
learning_rate = 0.1
# イテレーション数を算出
iteration_per_epoch = max(train_size / batch_size, 1)
iteration_num = iteration_per_epoch * epoch

# 定義したネットワークをインスタンス化
network = TwoLayerNetwork()

# 損失関数の出力結果を保持するリストを定義
train_loss_list = []
# 訓練画像の正解率を保持するリストを定義
train_accuracy_list = []
# テスト画像の正解率を保持するリストを定義
test_accuracy_list = []

print("学習を開始します.")

# 現在のエポック数をカウントする変数を定義
epoch_count = 0

# iteration_num回分、学習を繰り返す
for i in range(iteration_num):
    # バッチサイズ分の訓練画像をランダムに選択
    mask = np.random.choice(train_size, batch_size)
    image_batch = train_image[mask]
    label_batch = train_label[mask]

    # 勾配を計算
    gradient = network.calculate_gradient(image_batch, label_batch)

    # 各層のパラメータ(重みとバイアス)を更新
    for key in ('w_1', 'b_1', 'w_2', 'b_2'):
        network.parameters[key] -= learning_rate * gradient[key]
    
    # 損失を計算
    loss = network.calculate_loss(image_batch, label_batch)
    
    # エポックごとに正解率と損失を保存
    if i % iteration_per_epoch == 0:
        # エポック数をカウント
        epoch_count += 1
        # 訓練データの正解率を計算
        train_accuracy = network.calculate_accuracy(train_image, train_label)
        # テストデータの正解率を計算
        test_accuracy = network.calculate_accuracy(test_image, test_label)
        # 訓練データの正解率をリストに追加
        train_accuracy_list.append(train_accuracy)
        # テストデータの正解率をリストに追加
        test_accuracy_list.append(test_accuracy)
        # 損失をリストに追加
        train_loss_list.append(loss)
        print("------------------------------------------------")
        print("エポック数(epoch): {0}/{1} | 訓練データの精度(train accuracy): {2:.3f}, テストデータの精度(test accuracy): {3:.3f}".format(epoch_count, epoch, train_accuracy, test_accuracy))
        

print("学習が終了しました.")

# 学習終了時のパラメータを保存
with open('mnist.weights', 'wb') as web:
    pickle.dump(network.parameters, web)
print("学習結果をmnist.weightsに保存します.")


# 精度のグラフを描画
plt.subplot(2, 1, 1)
x = np.arange(len(train_accuracy_list))
plt.plot(x, train_accuracy_list, label='train accuracy')
plt.plot(x, test_accuracy_list, label='test accuracy', linestyle='--')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()