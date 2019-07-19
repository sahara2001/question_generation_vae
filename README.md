19/07/19

credit to bojone@github for base model and poem

### VAE问题生成模型

#### 使用
1. 环境: shi@六机:python3.6, tensorflow 1.14, keras自动装的, numpy　其他自己找
2. 运行：`python train.py`　模型在model.py，　一些自定义块在modules 里，数据来源ｎｌｐｃｃ２０１６kbqa
3. 数据在 train.txt中，也可以`python preprocessor.py`　处理nlpcc数据生成train.txt

#### 简介
用多层gcnn作为编码层把mask过后的文本输入映射为正态分布的表示，　再用类似结构的解码层重构原问题。　
网络以传统vae loss= 概率分布期望loss +　kl 散度 loss 作为优化目标，　以此学习字词的关系达到能提取出问题成分信息并以此生成与特定成分相关问题的最终目标（需要修改网络结构）。


#### 缺陷
1. 不确定mask有没有覆盖loss
2. 由于sequence不像图像，没什么连续性，很容易训练到一半挂了无穷大 loss.（或者搞个word2img类似ＤＦＴ让文字平滑,这样可以直接上DenseNet最后再inverse DFT一下)
3. 最终目标的ｌｏｓｓ的设计还没有头绪,
4. 要实现提取解析关键词的话，解码网络可能还得加上一些attention，用copy mechanism的话可能会比较容易实现
5. 网络不是传统的language model类型,复用同一个解码block(e.g. gcnn　竖着叠和rnn类似)会有改善, 但可能网络宽度hidden_dim得宽些加memory block来存信息

#### Todo
- [ ] 处理数据，提取问题中与答案有关的成分作为最终目标的训练集
- [ ] 尝试优化方法和更平滑的loss能使网络稳定学习
- [ ] 尝试在输入句子中挖掉和答案有关的关键词，再将关键词单独输入来‘特训’生成网络，再回去训练一个输入完整问题的生成器，从而避免新目标难收敛的问题


