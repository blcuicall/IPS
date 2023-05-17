# IPS

## 环境配置
版本requirements.txt: 
fairseq 1.0.0a0+01576be
pytorch 1.7.1
...

## 数据处理
在数据集文件夹News中处理数据
a.单语语料切分成train.txt , valid.txt, test.txt
b.提取关键词（yake），运行News/mask_data.py，生成的文件在multi_mask_data中
c.生成BPE的词表，利用fastbpe在multi_mask_data中学习bpe词表,得到code.bpe文件

./fast learnbpe 32000 train.txt > code.bpe

其中./fast为fastbpe安装路径中的fast
d.运行News/make_span_data.py生成片段预测的数据，生成的文件在span_data中，需要提前新建span_data/bpe文件夹。注意需要分别运行三次来处理train\valid\test，具体修改make_span_data.py中的文件名 所有train改成valid/test即可
e.在span_data下新建processed文件夹，将bpe文件夹下的数据处理成二进制文件，具体运行bash data_process.sh。提前在span_data文件夹下新建processed文件夹

## 模型训练
训练模型使用的是processed中的数据。训练脚本 train_news.sh

## 模型预测
evaluate.sh
batch_span_predictor_news.py为并行预测脚本
