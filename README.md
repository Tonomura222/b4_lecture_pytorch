# B4輪講　PyTorch

このサイトの0 PyTorch入門 [8]クイックスタートをもとに作りました  
https://yutaroogawa.github.io/pytorch_tutorials_jp/

`tutorial.ipynb`はサイトのコードと同じ

## validation追加
コードは`tutorial_v.ipynb`  
学習用のうち10000枚を検証用に使用

各epochで検証データに対する損失を求め、その損失が最小のepochのモデルでテストを行う  
以降はすべてvalidationあり

## CNNモデル
`tutorial_cnn.ipynb`
モデルを畳み込みに変更

## ResNetモデル
`tutorial_resnet.ipynb`
モデルをResNet-18に変更  
NN, CNNとのテスト結果の比較も行う


