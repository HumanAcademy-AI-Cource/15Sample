このプログラムはヒューマンアカデミー株式会社のAI入門講座で使用するプログラムです。

# Mnist Sample
Mnist(手書き数字のデータセット)を使い、Numpyのみで学習と予測をするパッケージ

# Requirements
- numpy
- pickle
- matplotlib

# Setup
## Matplotlibのインストール
```sh
./setup.sh
```
## Mnistのダウンロード
```sh
./download_mnist.sh
```
- 実行後、以下のリンクからMnistを`dataset`フォルダに保存されます。
    - [Mnistのダウンロード](http://yann.lecun.com/exdb/mnist/)

# Usage
## 訓練データの画像を表示
```sh
python show_mnist.py
```
- 実行後、訓練データからランダムに20枚の画像を選び、表示します。
## 学習
```sh
python train.py
```
- 実行後、学習が開始されます。
- 学習中は以下のパラメータが表示されます。
    - 現在のエポック数
    - 訓練データの正解率
    - テストデータの正解率
- 学習が終了するとモデルの精度を示すグラフが表示され、モデルの各パラメータが`mnist.weights`という名前のファイルで保存されます。
## 予測
### テストデータから予測
```sh
python predict_test.py
```
- 実行後、`mnist.weights`を読み込み、入力したテストデータの画像を表示します。
    - `w`キーを入力すると予測が開始されます。
    - `q`キーを入力すると終了します。

### 訓練データから予測
```sh
python predict_train.py
```
- 実行後、`mnist.weights`を読み込み、入力した訓練データの画像を表示します。
    - `w`キーを入力すると予測が開始されます。
    - `q`キーを入力すると終了します。
### Case1: データ数と学習回数が十分であり、予測精度が約96%
```sh
python case1_predict.py
```
### Case2: データ数が少なく、予測精度が約33%
```sh
python case2_predict.py
```
### Case3: 学習回数が少なく、予測精度が約78%
```sh
python case3_predict.py
```
