## 麻将大作业

### 任务队列

1. DataSet和Model管理(卲奕佳)
    1. 扩展Feature (mahjong/feature.py, mahjong/agent.py, mahjong/feature_default.py)
    2. 开发一个本地测评agent能力的application (mahjong/model.py)
        - 由于RL的应用是自对弈，没有一个很好的方式来知道模型好坏，需要通过模拟对战来发现一个最好的model;
        - 实现 mahjong/model.py/ModelManager::get_best_model
        - 对于模拟对局可以参考mahjong/actor.py

2. Supervised 部分 (朱越)
    1. 解决过拟合 (train_supervised.py, mahjong/model.py),提升正确性(目前16轮只有40%acc)

3. RL (陈思元)
    1. 与Supervised 部分接通 (完成)
    2. 开始训练

4. Botzone和可视化调试（吴泉霖）
    1. 上传整个文件, 程序入口点在__main__.py
    2. 可以先尝试在本地跑通，测试输入在wiki上有;

### Note
1. 运行指令
   ```shell
   cp -r /code/* /workspace && cp /dataset/* /workspace/data && cd /workspace && python3 preprocess.py && python3 supervised.py
   ```

2. 本地测试：
    ```sh
    # 搭建环境,在linux下运行
    conda create -n mahjong python=3.8 
    conda activate mahjong
    pip install torch MahjongGB multiprocessing
    source setup.sh 
    
    # 训练
    python ./preprocess.py
    python ./train_supervised.py
    python ./train_rl.py
    ```

3. 代码结构
```
.
├── data
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
│       ├── dataset.py
│       ├── __init__.py
│       ├── preprocess.py
│       └── supervised.py
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
