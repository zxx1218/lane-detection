# 计算机毕业设计-基于深度学习的车道线检测算法设计与实现
**我的CSDN中还有其他方向的深度学习毕业设计项目，例如图像破损修复，照片色彩增强，划痕检测，视频异常检测、车牌识别、目标检测等，具体参考**
[深度学习方向毕业设计](https://blog.csdn.net/qq_45566099/category_12507289.html)

## :sparkles: 效果展示！
![车道线检测](https://img-blog.csdnimg.cn/direct/b2e30fead8f34308820e390f69880a80.jpeg#pic_center)
## :sparkles: 在实时视频中进行车道线检测！


https://www.bilibili.com/video/BV1TH4y1P7vq/?spm_id_from=333.999.0.0

https://www.bilibili.com/video/BV1qt421A7wG/?spm_id_from=333.999.0.0


<hr>

#### 介绍
&emsp;&emsp;车道是智能汽车视觉导航系统的关键。车道自然是一个具有高级语义的交通标志，但它具有特定的局部模式，需要详细的底层特征才能准确定位。使用不同的特征级别对于准确的车道检测非常重要，但目前还没有得到充分的研究。在这份项目中，使用CNN跨层细化算法构建网络，旨在充分利用车道检测中的高层和低层特征。模型对车道进行检测的过程中，首先检测具有高级语义特征的车道，然后根据低级特征进行细化。通过这种方式，模型可以利用更多的视频中车道的上下文信息来检测车道的准确位置，同时利用详细的车道特征来提高定位精度。该算法使用了Vision Transformer来收集全局上下文，从而进一步增强了车道的特征表示。对于模型的梯度更新，算法引入了IoU损耗，将车道线路作为一个整体进行回归计算，从而提高车道线的定位精度。

<hr>


## 通过搭建前后端Web页面实现视频上传与自动检测
- **Web端演示视频**
- https://www.bilibili.com/video/BV1qt421A7wG/?spm_id_from=333.999.0.0


- **Web端体验地址：（稍后部署后更新）**


## :rocket: 使用方式
### 环境配置

- Python >= 3.8
- PyTorch >= 1.6
- CUDA
- 显存大于等于8G的英伟达显卡一张（如果要训练，那么需要一张最好是24G显存的显卡,如果没有显卡，作者提供GPU服务器租赁服务，每月100rmb）

具体操作：

```shell
# cd到代码根目录
cd 根目录
# 安装pytorch，注意如果你的显卡是3090、4090这种算例在官方8.0以上的，不能用10.x的Cuda，需要至少11.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# 安装依赖
pip install -r requirements.txt
```

### 数据集准备

CULane数据集（[CULane 数据集 (xingangpan.github.io)](https://xingangpan.github.io/projects/CULane.html)）

- 下载 CULane数据集，然后创建data文件，并将它们解压到data文件夹下

```shell
cd 代码根目录
mkdir -p data
# 下载解压
```

CULane解压后的结构是这样的：

```shell
$CULANEROOT/driver_xx_xxframe    # 6个数据文件
$CULANEROOT/laneseg_label_w16    # 车道线分割标签
$CULANEROOT/list                 # 数据集列表
```

### 调用模型进行测试（一定请确保前面的步骤都做好了）

对于测试，请运行：

```shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num] --view
```

参数解释：

```
1. [configs/path_to_your_config]  --->  数据集的配置文件，如果你用CULane数据集，这个配置文件在configs/clrnet/下
2. --[test|validate]  --->  选择进行验证还是执行测试(分别对应不同的模型文件)
3. --load_from [path_to_your_model]  --->  根据前一步--[test|validate]的选择结果，指定.pth文件（训练好的模型文件）的存放路径
4. --gpus  --->  选择测试所使用的显卡，0代表第一张，1代表第二张，以此类推，如果你只有一张卡，那么写0即可
5. --view ---> 可视化结果
```
结果会自动保存在./work_dirs/visualization下


## 通过部署Web端网页服务
cd到代码根目录下的gradioDemo文件夹下，然后执行python脚本（需要提前安装gradio包）
```
# 安装gradio包
pip install gradio 
# 启动Web服务
python gradio_demo.py
```
回车后会自动启动Web服务,默认启动端口为9091，在浏览器输入http://127.0.0.1:9091即可访问，在控制台看到如下信息代表成功启动👇
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c1a73a1a7d1f4f0091ec35b5215070c1.png#pic_center)
打开http://127.0.0.1:9091，显示如下界面👇
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/53271b7b66524ea5ad7460102e7e462f.png#pic_center)


<hr>

## :pencil2:	如何自己训练模型?

- **咨询作者**

## 有问题联系作者：
- VX：Accddvva
- QQ：1144968929
- 该项目在github与gitee上提供训练好的模型文件以及调用该文件的测试代码，clone后安装环境即可使用
- **本项目完整代码（测试+训练+模型定义）+ 环境配置教程 + 代码使用方式 ==> 价格300RMB，可提供远程部署服务，另外没有合适的显卡的同学可提供GPU服务器短期租赁服务，24G显存服务器每个月100RMB**

<hr>

#### :fire:广告
- 作者于浙江某985高校就读人工智能方向研究生，可以帮忙定制设计模型，并提供源代码和训练后的模型文件以及环境配置和使用方法，只需要描述需求即可。
- 人工智能领域，尤其是计算机视觉（Computer vision，CV）方向的毕业设计，只要你想得出，没有做不出的
