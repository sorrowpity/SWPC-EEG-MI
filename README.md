SWPC
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
一、准备环境：
为了确保万无一失，请向接手人提供以下完整的操作流程：创建和激活环境：

conda create -n bci_env python=3.10

conda activate bci_env
下载/放置代码和数据：

将所有代码文件（train_swpc.py, ShallowConvNet.py 等）和 requirements.txt 文件放置在项目根目录。
将数据文件（train_A*.mat 等） 放置在 ./data/ 目录下。

数据文件下载位置
https://pan.baidu.com/s/14-f08DyaTVIuOKdsEY89zw 提取码：6uem

安装 PyTorch (关键步骤)：
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

安装其他依赖：
pip install -r requirements.txt

验证环境：
python -c "import torch; print(f'Torch Version: {torch.version}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

二、开始训练：
运行
python train_swpc.py。

这个脚本会自动调用你的 Loader，分别训练“预筛选”和“分类”两个模型。

跑完后，当前目录下会出现 prescreen_model.pth 和 classifier_model.pth。这就是老师要的结果。

测试控制：

通过修改inference_robot.py中的变量DATA_SOURCE_MODE和DEPLOY_MODE，可以自主选择是否采用脑机接口设备模拟输入和模拟机械臂输出

DEPLOY_MODE = True
True:发送指令到机械臂； 与mock_robot_server.py搭配使用
False: 仅在本地打印指令

DATA_SOURCE_MODE = "REAL_DATASET"
可选项：
"REAL_DATASET"  data中的内容,
"LIVE_SOCKET" 模拟真实脑际设备输入，

与mock_eeg_source.py搭配使用
如果使用True和LIVE_SOCKET则在终端输入命令分别运行输入和输出以及处理程序

(需要三个终端)
参考如下：
1首先进入目录下：
cd SWPC
2开启输出
python mock_robot_server.py
3开启输入
python mock_eeg_source.py
4开启处理
python inference_robot.py
如果看到控制台输出“指令：右手...”，说明你的整个异步脑控链路已经打通了！

另有一個4SWPC的文件夾，是4分類的任務，現在還尚未完成。。。