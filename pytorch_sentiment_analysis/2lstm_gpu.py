#!/usr/bin/env python
# coding: utf-8

# 同样使用RNN，本节的改进如下
# 
# * 预训练的词向量
# * 不同的RNN结构（LSTM）
# * 多层RNN
# * 正则化
# * 不同的优化器

# ---

# ## 1. 数据准备
# 
# 通过预训练词向量初始化每个词

# In[3]:


import torch
from torchtext import data, datasets
import random
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


SEED = 1234
begin_t = time.time()

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print('prepare data time:', time.time() - begin_t)


# In[4]:


TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

print('build vocabulary time:', time.time() - begin_t)


# In[5]:


BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)


# ## 2. 构建模型
# 
# * 使用复杂的RNN模型
# * 使用Dropout防止过拟合

# In[6]:


import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN的结构，1. 输入x的维度，2. 隐藏层的维度，3. RNN的层数，4. 是否是双向的， 5. 层与层之间的dropout比率
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        
        # rnn的输出最终会将前向和后向的最后隐藏层状态拼接起来，所以输入到线性全连接层时，维度是hidden_dim的两倍
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # dropout层在forward中使用，应用到我们想要dropout的层
        # dropout永远不应该在输入层和输出层使用
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]
        # 不是说输入层不能用dropout吗？是为了避免预训练的词向量特性太强？
        embedded = self.dropout(self.embedding(x))
        # embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        
        # hidden[-2, :, :] --> forward_layer_n
        # hidden[-1, :, :] --> backward_layer_n
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        
        # hidden = [1, batch size, hid dim * num directions]  ?  
        return self.fc(hidden.squeeze(0))


# In[7]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


# In[8]:


pretrained_embedding = TEXT.vocab.vectors
print(pretrained_embedding.shape)


# In[9]:


model.embedding.weight.data.copy_(pretrained_embedding)


# ## 3. 训练（与评估）

# In[10]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 把模型放到GPU上内存不够
model = model.to(device)
criterion = criterion.to(device)


# In[11]:


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train() # 必须包含，以确保开启"dropout"
    
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()  # 求参数梯度
        optimizer.step()  # 更新参数权重
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()  # 必须包含，确保"dropout"关闭
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    begin_t = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print('Epoch: %02d, Train loss: %.3f, Train acc: %.2f, Valid loss: %.3f, Valid acc: %.2f' %
          (epoch, train_loss, train_acc, valid_loss, valid_acc), time.time() - begin_t)


# In[ ]:


test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('Test loss: %.3f, Test acc: %.2f' % (test_loss, test_acc))


# ## 使用

# In[ ]:


import spacy
nlp = spacy.load('en')


def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)] # 分词
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]  # 转化为下标
    tensor = torch.LongTensor(indexed).to(device)  # python list转换成pytorch tensor
    tensor = tensor.unsqueeze(1)  # [1, 2, 3] -> [[1, 2, 3]]，将单个样例转换为batch大小为1的batch
    prediction = torch.sigmoid(model(tensor))  # 预测值，并使用sigmoid挤压
    return prediction.item()  # 获取单个数字


# In[ ]:

review1 = 'This film is terrible'
print(review1, predict_sentiment(review1))


# In[ ]:

review2 = 'This film is stupid'
print(review2, predict_sentiment(review2))


# In[ ]:

review3 = 'This film is awesome'
print(review3, predict_sentiment(review3))

