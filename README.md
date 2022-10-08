# 方案介绍

微信大数据挑战赛2022初赛代码，复赛拉胯了（没考虑clip提取图像特征）加上复赛要求在云服务器上开发，所以代码懒得整理了，初赛思路大概是模型上采用了一个双塔模型，title，asr，ocr各取了128长度分别送入bert提取特征，视频特征通过了一个一层的transformer进一步提取视频帧间的交互特征，最后title，asr，ocr和视频的特征拼接送入一个3层transformer提取文本和视频交互特征，考虑到文本特征存在缺失还有噪声比较多，所以最后只提取视频特征部分做mean pooling分类。在训练时首先是利用无标签数据和有标签数据做预训练，预训练采用的是对比学习+moco的方式，预训练完后再用有标签数据做微调。初赛单模单折693（b榜），五折699（b榜），复赛的话就是用swin base提图像特征，16帧，fp16预测，其他和初赛一样，单模单折a榜707，b榜709，

主要代码都在src中

其中dataset中数据处理部分，视频特征读取参考了部分baseline代码

moco的部分实现参考自[GitHub - facebookresearch/moco: PyTorch implementation of MoCo: https://arxiv.org/abs/1911.05722](https://github.com/facebookresearch/moco)

bert采用的是Chinese-roberta-wwm-ext，对应src中Chinese-roberta-wwm-ext文件夹，文件中没有包含roberta的预训练权重，预训练权重需要通过[hfl/chinese-roberta-wwm-ext · Hugging Face](https://huggingface.co/hfl/chinese-roberta-wwm-ext)下载



# 环境依赖

python==3.7.11

pytorch==1.9.0

transformers==4.17.0

cuda10.2

# 训练

训练分为三步

1、预训练 

python ./src/pretrain_moco.py



2、生成5折训练、验证集

python ./src/ make_train_test_cross.py



3、训练5折模型

python ./src/train_cross.py



可以通过分别运行上面三步的脚步训练，也可以直接通过sh train.sh训练



# 预测

直接运行

sh inference.sh

预测结束后在data 目录下会生成一个result.csv的结果文件