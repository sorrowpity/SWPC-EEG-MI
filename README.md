# SWPC
Algo about self-supervised learning for MI

使用说明：在真实场景下，推荐使用CSP&FBCSP来判断当前是否处于静息态或运动想象态，EEGNet的性能或许更高，但是不稳定。而在明确当前为运动想象态的前提下，推荐使用正文中所说 EEGNET先supervised-learning再接self-supervised learning来分类，会取得更好的效果。

而EEGNET先SL再接SSL的方法在跨被试场景下会有明显更好的性能，可以与EA结合，但是在被试内场景下性能不够稳定。总结如下：CSP结合EEGNet效果更好，跨被试场景下EEGNET先SL再接SSL性能提升非常明显。

1.最基础的版本：被试内场景下的matlab代码的SWPC框架,请运行 fix_within_new.m  数据集放在 百度网盘： 链接：https://pan.baidu.com/s/14-f08DyaTVIuOKdsEY89zw 提取码：6uem。目前已经优化完代码。

2.跨被试场景下的matlab代码的SWPC框架,请运行 fix_within_cross.m 其中可以通过调整 EA = true/false 来控制使用不使用EA

3.跨被试或被试内场景python代码+supervised learning + self-supervised learning

loss.py 自监督学习涉及的LOSS函数

pre_train_cross_subject.py 预训练深度学习模型的代码

self_supervised_cross_subject.py 自监督学习需要用的代码

2025-12-06 张津铭留，以上是原作者的内容


总结与操作指南
准备环境： 确保你安装了 torch, numpy, scipy。

更新 Model： 复制我给的 EEGNet 代码，更新你的 model.py。

开始训练： 运行 python train_swpc.py。

这个脚本会自动调用你的 Loader，分别训练“预筛选”和“分类”两个模型。

跑完后，当前目录下会出现 prescreen_model.pth 和 classifier_model.pth。这就是老师要的结果。

测试控制： 运行 python inference_robot.py。

如果看到控制台输出“指令：右手...”，说明你的整个异步脑控链路已经打通了！

接下来的步骤就是将 inference_robot.py 中的 fake_data 替换为真实的脑电采集设备接口数据。