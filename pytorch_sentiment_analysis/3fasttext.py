#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 对每一个句子生成2-gram，用来捕获局部信息，这些2-gram直接添加到句子分词词序列的最后

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

generate_bigrams(['This', 'film', 'is', 'terrible'])


# In[5]:


import torch
from torchtext import data, datasets
import time

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deteriministic = True

begin_t = time.time()
# preprocessing指明一个操作，该操作执行在分词之后（string变成list），但是在indexed之前???(indexed会对二元组进行？？？)
#TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)



# In[6]:


import random

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))


# In[7]:


TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

print('prepare data time:', time.time() - begin_t)


# In[8]:


BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)


# In[11]:


print(vars(train_data.examples[0]))


# ## 模型
# 1. 平均句子中每个词的向量（使用avg_pool2d）
# 2. 送入Linear层

# In[10]:


import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x):
        # x = [sent len, batch size]
        embedded = self.embedding(x)
        # embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2)
        # embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # pooled = [batch size, emb dim]
        return self.fc(pooled)


# In[12]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)


# In[18]:


pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)


# ## 训练

# In[14]:


import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# In[15]:


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
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
    print('%02d Train loss: %.3f, Train acc: %.2f%%, Valid loss: %.3f, Valid acc: %.2f%%'           % (epoch, train_loss, 100*train_acc, valid_loss, 100*valid_acc), time.time() - begin_t)


# In[ ]:


test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('Test loss: %.3f, Test acc: %.2f%%' % (test_loss, 100*test_acc))


# ## USE

# In[16]:


import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


# In[ ]:


print(predict_sentiment('This film is awesome'))
print(predict_sentiment('This file is great'))
print(predict_sentiment('This file is a shit'))

