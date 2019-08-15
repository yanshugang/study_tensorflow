data 文件夹里的是训练和测试所用的 PTB 数据集。

=======
训练：

运行 python train.py 即可开始训练。
也可以使用参数 --data_path 来指定所使用的数据集的目录，
一般来说使用我提供的 data 目录即可，不需要用 --data_path 参数。

=======
测试：

训练完很多个 Epoch（最终精度达到大概 30% 以上）之后，
将训练时生成的 train-checkpoint-* 参数文件之一改名
（名字匹配 utils.py 中的 load_file），
运行 python generate.py 即可生成图片。

也可以使用参数 --load_file 来指定要用哪个参数文件来测试，
例如：
python generate.py --load_file train-checkpoint-67

具体使用请看本课程的视频和参考 utils.py 文件。
