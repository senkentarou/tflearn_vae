# tflearnによるVAEの構成

## 概要

 * tflearnでVAE(Variational AutoEncoder)を実装
 * データセットはMNIST

## 内容物

 * scripts/train_vae_mnist.py: 学習用
 * scripts/evaluate_gan_mnist.py: 評価用(画像生成)
 * results/mnist_trained_50000_epoch_100_hidden_256.png
   (各中間層数256, 学習サンプル50000, エポック数100のときの結果)
 
## 環境

 * ubuntu 14.04 LTS
 * python 3.4.5 (virtualenv)
   * tensorflow >= 1.0.0 [Installing Tensorflow on Ubuntu](https://www.tensorflow.org/install/install_linux)
   * tflearn >= 0.3.0 [tflearn - github](https://github.com/tflearn/tflearn)

## 実行手順

 1. train_vae_mnist.pyを実行
  * 初回実行時は, mnistが../datasets/mnistにダウンロードされる.
 2. evaluate_vae_mnist.pyを実行
  * resultsディレクトリにresult.pngが生成されていることを確認する.

## その他
 * 潜在変数の次元数を10としている.
 * 区分線形ユニットや, 中間層の数, エポック数などはハイパーパラメータとして決定したため,  
   適切でない可能性がある.
