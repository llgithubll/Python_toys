## word2vec

word2vec in python3(and numpy)

本程序使用Python3.6和Numpy实现了基于Negative Sampling的Skip-gram词向量模型，并且集成了Subsampling技术，同时还包括对低频词的裁剪工作；最后利用余弦相似，求得与输入词汇最相似的10个词汇，来检验词向量模型训练的效果。

程序代码如下所示，程序本身有很强的自解释性，没有过多的注释；算法的核心内容（参数更新，行188-行200）可以参考上述理论中手写的公式推导和伪代码；算法中的W1,W2两个权重矩阵分别为上文所述的W,W'，前者是每个词看做中心词的向量表示，后者是每个词作为上下文词的向量表示。

算法使用的训练数据集，可通过以下方式获取（或直接通过链接下载，然后解压），训练的词向量会保存在文本文件中（text8大小为100M，普通PC训练时间可能要一两个小时）：

``` 
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f
```


参(zhao)考(chao)了[this](https://github.com/tscheepers/word2vec/blob/master/word2vec.py)

