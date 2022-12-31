## 麻将大作业

### 任务队列

1. DataSet和Model管理(卲奕佳)
    1. 扩展Feature (mahjong/feature.py, mahjong/agent.py, mahjong/feature_default.py)
    2. 开发一个本地测评agent能力的application (mahjong/model.py)
        - 由于RL的应用是自对弈，没有一个很好的方式来知道模型好坏，需要通过模拟对战来发现一个最好的model;
        - 实现 mahjong/model.py/ModelManager::get_best_model
        - 对于模拟对局可以参考mahjong/actor.py

2. Supervised 部分 (吴泉霖)
    1. 解决过拟合 (train_supervised.py, mahjong/model.py)
        - 可变lr：已完成
        - Batch Normalization：TODO
        - 数据增广：由于绿一色的存在，只能增广万和筒
    2. 目前train:98%, val:76%

3. RL (陈思元)
    1. 与Supervised 部分接通 (完成)
    2. 开始训练
    3. 训练算法:
    ```python 
    model = SupervisedModel()
    while True:
        episode = run_episode(model, model, model, model)
        model.update(episode)
    ```

    ```python 
    model = baseline = SupervisedModel()
    while True:
        episode = run_episode(model, baseline, baseline, baseline)
        model.update(episode)
        if model is better than baseline:
            baseline = model
    ```
4. Botzone和可视化调试（吴越）
    1. 负责botzone比赛，
    2. 负责接通将model file放到botzone上
    3. 负责从botzone观察反馈；
    2. 上传整个文件, 程序入口点在__main__.py
    3. 大家可以都传一个bot启动天梯排行（操作下面有），测评次数更多

### Note
1. 运行指令
<<<<<<< HEAD
   ```shell
   cp -r /code/* /workspace && cp /dataset/* /workspace/data/data && cd /workspace && python3 preprocess.py && python3 train_supervised.py
   ```

2. 本地测试：
    ```sh
    # 搭建环境,在linux下运行
    conda create -n mahjong python=3.8
    conda activate mahjong
    pip install torch PyMahjongGB multiprocessing
    source setup.sh

    # 训练
    python ./preprocess.py
    python ./train_supervised.py
    python ./train_rl.py
    ```

3. 代码结构
```
.
├── new_data
│   ├── *.npz
│   ├── count.json
│   └── data
│       ├── data.txt
│       ├── README-en.txt
│       ├── README.txt
│       └── sample.txt
├── gitignore
├── mahjong
│   ├── agent.py
│   ├── feature_default.py
│   ├── feature.py
│   ├── __init__.py
│   ├── model.py
│   ├── RL
│   │   ├── actor.py
│   │   ├── env.py
│   │   ├── __init__.py
│   │   ├── learner.py
│   │   ├── model-pool
│   │   ├── model_pool.py
│   │   ├── model.py
│   │   ├── __pycache__
│   │   ├── replay_buffer.py
│   │   └── train.py
│   └── Supervised
│       └── dataset.py
├── __main__.py
├── model
│   └── checkpoint
|           └── model_*.pt
├── preprocess.py
├── python
├── readme.md
├── setup.sh
├── train_rl.py
└── train_supervised.py

```

4. Botzone使用
    1. 参数上传
        - 在个人存储空间中上传模型文件 *.xxx
        - （在加载模型时指定文件路径为/data/*.xxx）
    2. 代码打包
        - 压缩_main_.py和mahjong子文件夹为.zip文件
    3. bot上传
        - 上传.zip文件为源代码
        - 编译器选python3
        - 勾选长时运行和简单交互
=======
    ```shell
    cp -r /code/* /workspace && cp /dataset/* /workspace/data && cd /workspace && python3 preprocess.py && python3 supervised.py
    ```
    注意notebook测试时：
    1. 如果上传的代码在压缩包内的子文件夹下，需要把code文件夹下的子文件夹展开
        ```shell
        cp -r /code/Supervised/* /code
        (rm -rf /code/Supervised)
        ```
    2. 需要在根路径下创建空model文件夹
        ```bash
        mkdir /model
        ```
    3. 第二次运行supervised.py之前需要删去model中的checkpoint子文件夹
        ```bash
        rm -rf /model/c*
        ```

### Supervised
改动：
> 使用了新的feature.py(增加到133维)和model.py（引入了ResNet特性）
>
> 修改了supervised.py(train acc & lr_scheduler & epoch)

效果：
> 见log.out，30个epoch有点过拟合？

### model
在notebook模式时不会返回模型（可以通过terminal cp到code后下载），是不是要用训练模式才行？

### botzone
调用模型遇到了一点问题，报错信息在debug.out里，完整日志在botzone里
>>>>>>> af0ef81915da4eb19706369ba40322c28efb98c2
