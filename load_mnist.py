#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリをインポート
import pickle
import numpy as np
import gzip


# 保存されているデータセットのファイル名を定義
load_file_name ={
    'train_image':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_image':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def load_label(file_name):
    """
    データセットからラベルを読み込む関数
    """

    # ファイルパスを指定
    file_path = 'dataset/' + file_name

    # ファイルを開き、ラベルデータを変数に代入
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # one-hot-label形式にラベルデータを変換
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    
    return one_hot_labels

def load_image(file_name):
    """
    データセットから画像を読み込む関数
    """

    # ファイルパスを指定
    file_path = 'dataset/' + file_name 
    # ファイルを開き、画像データを変数に代入
    with gzip.open(file_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    
    return images

def change_format(image_data):
    """
    画像データの形式に変換する関数
    """

    # ピクセル値の型を変更
    image_data = image_data.astype(np.float32)
    # ピクセル値の標準化
    image_data /= 255.0
    # 画像ごとにデータを分割
    image_data = image_data.reshape(-1, 28*28)
    
    return image_data

def load_dataset(rate, lower=0, upper=9):
    """
    任意の数と種類のデータセットを生成する関数
    """

    # 訓練画像を読み込む
    train_image = load_image(load_file_name['train_image'])
    # 訓練画像の形式を変更
    train_image = change_format(train_image)
    # 訓練画像のラベルを読み込む
    train_label = load_label(load_file_name['train_label'])
    # テスト画像を読み込む
    test_image = load_image(load_file_name['test_image'])
    # テスト画像の形式を変更
    test_image = change_format(test_image)
    # テスト画像のラベルを読み込む
    test_label = load_label(load_file_name['test_label'])

    # データセットを格納する変数を定義
    dataset = {}
    # 指定された数字のインデックスを取得
    train_mask = np.where((np.argmax(train_label, axis=1) >= lower) & (np.argmax(train_label, axis=1) <= upper))
    # 指定された数字をデータセットから切り取る
    train_label = train_label[train_mask][:]
    # 指定された割合のデータセットを切り取る
    dataset['train_label'] = train_label[:int(rate*train_label.shape[0])]

    # 指定された数字をデータセットから切り取る
    train_image = train_image[train_mask][:]
    # 指定された割合のデータセットを切り取る
    dataset['train_image'] = train_image[:int(rate*train_image.shape[0])]

    # 指定された数字のインデックスを取得
    test_mask = np.where((np.argmax(test_label, axis=1) >= lower) & (np.argmax(test_label, axis=1) <= upper))
    # 指定された数字をデータセットから切り取る
    test_label = test_label[test_mask][:]
    # 指定された割合のデータセットを切り取る
    dataset['test_label'] = test_label[:int(rate*test_label.shape[0])]
    
    # 指定された数字をデータセットから切り取る
    test_image = test_image[test_mask][:]
    # 指定された割合のデータセットを切り取る
    dataset['test_image'] = test_image[:int(rate*test_image.shape[0])]

    return dataset