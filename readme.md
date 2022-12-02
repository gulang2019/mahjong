## 麻将大作业

### 任务队列

1. 跑通代码 (陈思元)
    1. RL: 未完成
    2. Supervised: 已经开始训练
2. Preprocess 修改 (卲奕佳)
   目前只提取了6个feature, 希望提取更多的Feature
3. 完成网络 (RL: 吴泉霖, Supervised: 朱越)
    1. Embedding 取代 onehot
    2. 加深网络
4. 调试部分 (轮流调试, RL, Supervised并行)
    1. RL：卲奕佳，吴泉霖
    2. Supervised: 陈思元，朱越

### Note
1. 运行指令
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