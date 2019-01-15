#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torchtext import data, datasets
import random
import time
import os

begin_t = time.time()
SEED = 1234
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print('Load dataset time:', time.time() - begin_t)
print(f'Num of training examples:{len(train_data)}')
print(f'Num of valid examples:{len(valid_data)}')
print(f'Num of test examples:{len(test_data)}')


# In[ ]:


TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)
print(f'Unique tokens in TEXT vocabulary:{len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary:{len(LABEL.vocab)}')


# In[ ]:


BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [sent len, batch size]
        x = x.permute(1, 0)
        # x = [batch size, sent len]
        embedded = self.embedding(x)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conv_n = [batch size, n_filters, sent len - filter_size[n]]
        # conv detail: [N, Ci, W, D] -> [N, Co, W, 1] squeeze(3)-> [N, Co, W]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        # pool detail: [bat size, n_filters, sent len - filter_sizes[n]]
        #          --> [bat size, n_filters, 1] squeeze(2) -> [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filters_size)]
        return self.fc(cat)


# In[ ]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)


# In[ ]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return  acc


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


print('training...')
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    begin_t = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    print('Epoch %02d, Train loss: %.3f, Train acc: %.2f, Valid loss: %.3f, Valid acc: %.2f' % 
          (epoch, train_loss, train_acc, valid_loss, valid_acc), time.time() - begin_t)
    
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('Test loss: %.3f, Test acc: %.2f' % (test_loss, test_acc))


# In[ ]:


import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence, min_len=5):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

s1 = 'This film is terrible'
s2 = 'What a fuck film'
s3 = 'Wonderful feeling'

print(s1, predict_sentiment(s1))
print(s2, predict_sentiment(s2))
print(s3, predict_sentiment(s3))

