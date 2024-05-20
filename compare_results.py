# テスト結果の比較
import os

import pandas as pd
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

df_nn = pd.read_csv("result/test_nn.csv", index_col=0, header=0, names=["label", "pred_nn"])
df_cnn = pd.read_csv("result/test_cnn.csv", index_col=0, header=0, names=["label", "pred_cnn"])
df_cnn.drop(columns=["label"], inplace=True)
df_resnet = pd.read_csv("result/test_cnn.csv", index_col=0, header=0, names=["label", "pred_resnet"])
df_resnet.drop(columns=["label"], inplace=True)

df = pd.concat([df_nn, df_cnn, df_resnet], axis=1)
#nnだけ間違えた例
df_a = df[(df["label"] != df["pred_nn"]) & (df["label"] == df["pred_cnn"]) & (df["label"] == df["pred_resnet"])]
#resnetだけ正解した例
df_b = df[(df["label"] != df["pred_nn"]) & (df["label"] != df["pred_cnn"]) & (df["label"] == df["pred_resnet"])]

image_path = "result/image"
os.makedirs(image_path, exist_ok=True)

if len(df_a) != 0:
    index_a = df_a.index[0]
    print(df_a.iloc[0])
    image, label = test_data[index_a]
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("GT: {}".format(label))
    plt.savefig(os.path.join(image_path,"nn_failure.jpg"))

if len(df_b) != 0:
    index_b = df_b.index[0]
    print(df_b.iloc[0])
    image, label = test_data[index_b]
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("GT: {}".format(label))
    plt.savefig(os.path.join(image_path,"resnet_correct.jpg"))
