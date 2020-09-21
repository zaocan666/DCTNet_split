# DCTNet_split
基于DCTNet的高低频分离模型，分离高低频来减少运算量和信息量

## 文件夹
- rimage88_cat: 训练低频输入卷积部分和高频模型
- rimage90_catlow：训练低频其他部分

## 主要文件
- main/mini_imagenet_resnet_part.py:训练低频输入卷积部分和高频模型
- main/mini_imagenet_resnet_partlow.py:加载已训练参数，训练低频其他部分
- main/mini_imagenet_eval_part.py：在测试集中测试模型（当前文件夹下的模型参数）性能
- utils/calculate_flop.py：测量模型的参数量和运算量

## 训练
训练低频输入卷积部分和高频模型
 sh scripts/mini_resnet_part.sh
训练低频其他部分
 sh scripts/mini_resnet_partlow.sh

## 测试
python main/mini_imagenet_eval_part.py

## 安装所需库
* Install [PyTorch](http://pytorch.org/)

* Clone this repo recursively
  
* Install required packages
  ```
  pip install -r requirements.txt
  ```
  
* Install [libjpeg-turbo](http://www.linuxfromscratch.org/blfs/view/svn/general/libjpeg.html)
  ```
  bash install_libjpegturbo.sh

## citation
https://github.com/calmevtime/DCTNet
