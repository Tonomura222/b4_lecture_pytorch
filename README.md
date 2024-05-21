# B4輪講　PyTorch

基本はこのサイトの0 PyTorch入門 [8]クイックスタートをもとに作りました  
https://yutaroogawa.github.io/pytorch_tutorials_jp/

`tutorial.ipynb`はサイトのコードと同じ

## validation追加
コードは`tutorial_v.ipynb`  
学習用画像のうち10000枚を検証に使用し、検証のlossで最適epochを決定  
`tutorial_v.py`はそのままつなげてpythonファイルにしたもの
pythonかpython3で実行  
```
python3 tutorial_v.py
```

## CNNモデル
`tutorial_cnn.ipynb`, `tutorial_cnn.py`
モデルを畳み込みに変更

## ResNetモデル
`tutorial_resnet.ipynb`, `tutorial_resnet.py`
モデルをResNet-18に変更  

3つのテスト結果のcsvがあれば`compare_results.py` を実行して比較
```
python3 compare_results.py
```

## ペットのセグメンテーション
`oxfordpet_segmentation.ipynb`, `oxfordpet_segmentation.py`  
データセットはOxford IIIT Pets Segmentation  

このサイトを参考にしました。
https://www.kaggle.com/code/dhruv4930/oxford-iiit-pets-segmentation-using-pytorch

入力画像  
![alt text](image.png)  
正解ラベル  
![alt text](image-1.png)

```
python3 oxfordpet_segmentation.py
```

- epochごとのvalidationの予測結果
- テストの最良の例
- 損失のグラフ
などが保存されるはず
