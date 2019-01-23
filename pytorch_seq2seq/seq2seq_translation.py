"""
seq2seq + attention
"""

import os
import re
import random
import unicodedata
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
HAS_ATTENTION = True

"""
加载数据
"""


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # include 'SOS' and 'EOS'

    def add_sentence(self, sentence):
        for word in sentence.split():  # 句子中的单词默认以空格隔开
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    """法语具有和英语相同体系的字母表, 但法语的单词包含'声调', 可以转换掉, 以简化过程"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)  # 把标点[.!?]和单词以空格隔开
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 所有非字幕和[.!?]的符号直接用空格替换
    return s


def read_langs(lang1, lang2, reverse=False):
    """

    :param lang1:
    :param lang2:
    :param reverse: 用来转换翻译角色的: 英语->法语, or, 法语->英语
    :return:
    """
    print('Reading lines....')

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


"""
筛选简单句子子集, 先进行快速的run起来
"""

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False, do_filter=True):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if do_filter:
        pairs = filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)
print(random.choice(pairs))


"""
Model: seq2seq
"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # input_size是词典大小, hidden_size是词向量维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 这里nn.GRU(x, h)两个参数指明输入x和隐藏层状态的维度, 这里都用hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        :param input: 这里的input是每次一个词, 具体形式为: [word_idx]
        :param hidden:
        :return:
        """
        # input: [1]
        # embedding(input): [1, emb_dim]
        # embedded: [1, 1, 1 * emb_dim]
        embedded = self.embedding(input).view(1, 1, -1)

        # 关于gru的输入输出参数
        # [seq_len, batch_size, feture_size]
        # output: [1, 1, 1 * emb_dim]
        output = embedded
        # hidden: [1, 1, hidden_size]
        # 这里hidden_size == emb_dim
        output, hidden = self.gru(output, hidden)
        # output: [seq_len, batch, num_directions * hidden_size]
        # output: [1, 1, hidden_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(hidden_size, output_size)
        # softmax层, 求每一个单词的概率
        self.softmax = nn.LogSoftmax(dim=1)  # ?

    def forward(self, input, hidden):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        output = self.embedding(input).view(1, 1, -1)  # 展开
        # output: [1, 1, emb_dim]
        output = F.relu(output)
        # output: [1, 1, emb_dim]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        # output: [1, V] 值为每个单词的概率
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # * 2 = cat(embeding, hidden)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        embedded = self.embedding(input).view(1, 1, -1)  # 展开
        embedded = self.dropout(embedded)
        # embedded: [1, 1, emb_dim]

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)),
            dim=1
        )
        # attn_weights: [1, MAX_LENGTH]
        # encoder_outputs: [max_length, encoder.hidden_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # attn_applied: [1, 1, hidden_size]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # output: [1, hidden_size * 2]
        output = self.attn_combine(output).unsqueeze(0)
        # output: [1, 1, hidden_size]

        output = F.relu(output)
        # output: [1, 1, hidden_size]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # emb_dim == hidden_size
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        output = F.log_softmax(self.out(output[0]), dim=1)
        # output[1, V], 值为每个单词的概率
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



"""
训练: prepare training data
"""


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensor_from_pair(pair):
    """

    :param pair: ['ils sont fort joyeux .', 'they are very cheerful .']
    :return:
    """
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


pair = pairs[100]
p_test = tensor_from_pair(pair)
i_tensor = p_test[0]
t_tensor = p_test[1]
print(pair)
# print(i_tensor)
# print(t_tensor)
print('input_tensor:', i_tensor.shape, 'target_tensor:', t_tensor.shape)

# Train
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        # encoder 每次读取一个词, 重复input_length次
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_output: [1, 1, hidden_size]
        # encoder_output[ei]: [hidden_size]
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)#, encoder_outputs)
            # decoder_output: [1, V] 值为每个单词的概率
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        # without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)#, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_attention(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        # encoder 每次读取一个词, 重复input_length次
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_output: [1, 1, hidden_size]
        # encoder_output[ei]: [hidden_size]
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output: [1, V] 值为每个单词的概率
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        # without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

"""
计时工具
"""


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (as_minutes(s), as_minutes(rs))


def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, has_attention=True):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset each print_every
    plot_loss_total = 0  # reset each plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensor_from_pair(random.choice(pairs)) for i in range(n_iters)]
    # nn.NLLLoss(): The negative log likelihood loss. It is useful to train a classification problem with C classes.
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        if has_attention:
            loss = train_attention(
                    input_tensor, target_tensor,
                    encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    criterion)
        else:
            loss = train(input_tensor, target_tensor,
                         encoder, decoder,
                         encoder_optimizer, decoder_optimizer,
                         criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = print_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


"""
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
attention_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# train_iters(encoder1, decoder1, 75000, print_every=5000)
"""

"""
Reading lines....
Read 135842 sentence pairs
Trimmed to 10599 sentence pairs
Counting words...
Counted words:
fra 4345
eng 2803
['elle est habillee elegamment .', 'she s smartly dressed .']
['je pars .', 'i m going .']
input_tensor: torch.Size([4, 1]) target_tensor: torch.Size([5, 1])
13m 22s (-187m 14s) (5000 6%) 2.9315
27m 41s (-180m 2s) (10000 13%) 2.3854
41m 52s (-167m 30s) (15000 20%) 2.0639
55m 41s (-153m 9s) (20000 26%) 1.8147
69m 10s (-138m 21s) (25000 33%) 1.6102
82m 41s (-124m 2s) (30000 40%) 1.4127
96m 17s (-110m 2s) (35000 46%) 1.2860
109m 50s (-96m 6s) (40000 53%) 1.1307
123m 22s (-82m 14s) (45000 60%) 0.9979
137m 3s (-68m 31s) (50000 66%) 0.9042
150m 37s (-54m 46s) (55000 73%) 0.7839
164m 14s (-41m 3s) (60000 80%) 0.7183
177m 58s (-27m 22s) (65000 86%) 0.6539
191m 37s (-13m 41s) (70000 93%) 0.5675
205m 14s (-0m 0s) (75000 100%) 0.5193
"""


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            # encoder 每次读取一个词, 重复input_length次
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # encoder_output: [1, 1, hidden_size]
            # encoder_output[ei]: [hidden_size]
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)#, encoder_outputs)
            topv, topi = decoder_output.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()  # detach from history as input

        return decoded_words


def evaluate_attention(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            # encoder 每次读取一个词, 重复input_length次
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # encoder_output: [1, 1, hidden_size]
            # encoder_output[ei]: [hidden_size]
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()  # detach from history as input

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, n=30, has_attention=True):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if has_attention:
            output_words, attentions = evaluate_attention(encoder, decoder, pair[0])
        else:
            output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# evaluate_randomly(encoder1, decoder1)

r"""
Microsoft Windows [版本 10.0.17134.523]
(c) 2018 Microsoft Corporation。保留所有权利。

F:\Desktop\NLP>python seq2seq_translation.py
Reading lines....
Read 135842 sentence pairs
Trimmed to 10599 sentence pairs
Counting words...
Counted words:
fra 4345
eng 2803
['je suis content de mon emploi .', 'i am content with my job .']
['je pars .', 'i m going .']
input_tensor: torch.Size([4, 1]) target_tensor: torch.Size([5, 1])
13m 15s (-185m 33s) (5000 6%) 2.8940
26m 43s (-173m 44s) (10000 13%) 2.3405
40m 14s (-160m 58s) (15000 20%) 2.0321
53m 44s (-147m 47s) (20000 26%) 1.7967
67m 26s (-134m 53s) (25000 33%) 1.6020
81m 3s (-121m 35s) (30000 40%) 1.4133
94m 42s (-108m 14s) (35000 46%) 1.2297
108m 20s (-94m 47s) (40000 53%) 1.1133
122m 7s (-81m 24s) (45000 60%) 1.0101
135m 52s (-67m 56s) (50000 66%) 0.8732
149m 30s (-54m 21s) (55000 73%) 0.7792
163m 16s (-40m 49s) (60000 80%) 0.7111
177m 4s (-27m 14s) (65000 86%) 0.6418
190m 46s (-13m 37s) (70000 93%) 0.5751
204m 29s (-0m 0s) (75000 100%) 0.5222
> il est bon cuisinier .
= he is good at cooking .
< he is good at cook . <EOS>

> nous avons confiance .
= we re confident .
< we re confident . <EOS>

> il est tout sauf honnete .
= he is anything but honest .
< he is anything but honest . <EOS>

> c est bien .
= i m glad to hear that .
< i m fine . <EOS>

> je suis desolee . j ai oublie .
= i m sorry . i forgot .
< i m sorry . . <EOS>

> desole j ai perdu mon sang froid .
= i m sorry i lost my temper .
< i m sorry i lost my lost . <EOS>

> je m habitue a manger seul .
= i m getting used to eating alone .
< i m getting used to getting alone . <EOS>

> vous etes charmants .
= you re charming .
< you re charming . <EOS>

> tu es dedans jusqu au cou .
= you re in over your head .
< you re in over over over . <EOS>

> il est dote d un talent incroyable .
= he s incredibly talented .
< he s incredibly talented . <EOS>

> vous plaisantez bien sur .
= you re joking of course .
< you re joking of course . <EOS>

> je suis affame et assoiffe .
= i m hungry and thirsty .
< i m hungry and thirsty . <EOS>

> tu es fort en colere .
= you re very angry .
< you re very angry . <EOS>

> nous sommes depourvus de prejuges .
= we re unprejudiced .
< we re unprejudiced . <EOS>

> c est mon compagnon .
= he s my partner .
< he s my partner . <EOS>

> vous me mettez les mots dans la bouche .
= you re putting words in my mouth .
< you re never in my words in . <EOS>

> je suis juste un autre homme .
= i m just another man .
< i m just a man . <EOS>

> nous sommes amoureux .
= we re in love .
< we re in love . <EOS>

> c est une femme d affaire a succes .
= she s a successful businesswoman .
< she s a successful of . . <EOS>

> je suis ton plus grand fan .
= i m your biggest fan .
< i m your biggest fan . <EOS>

> nous sommes observes .
= we re being watched .
< we re being watched . <EOS>

> encore es tu plus grand que moi .
= you re still taller than me .
< you re taller than me . <EOS>

> je suis assez embrouillee comme ca .
= i m confused enough as it is .
< i m confused enough as it is . <EOS>

> tu n es pas cense nager ici .
= you aren t supposed to swim here .
< you aren t supposed to swim here . <EOS>

> je suis contre ce projet de loi .
= i m against the bill .
< i m against the bill . . <EOS>

> nous sommes piegees .
= we re trapped .
< we re trapped . <EOS>

> je vais mieux .
= i m feeling better .
< i am better . <EOS>

> elle est determinee a quitter l entreprise .
= she is determined to leave the company .
< she is determined to leave the . . <EOS>

> t es un gamin bizarre .
= you re a weird kid .
< you re a weird kid . <EOS>

> c est ici qu il joue .
= he is playing here .
< he is quite here . <EOS>

F:\Desktop\NLP>
"""

# ---------------------Attention---------------------------------------------


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

train_iters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluate_randomly(encoder1, attn_decoder1)

r"""
Microsoft Windows [版本 10.0.17134.523]
(c) 2018 Microsoft Corporation。保留所有权利。

F:\Desktop\NLP>python seq2seq_translation.py
Reading lines....
Read 135842 sentence pairs
Trimmed to 10599 sentence pairs
Counting words...
Counted words:
fra 4345
eng 2803
['elle est super .', 'she s awesome .']
['je pars .', 'i m going .']
input_tensor: torch.Size([4, 1]) target_tensor: torch.Size([5, 1])
14m 37s (-204m 45s) (5000 6%) 2.8207
29m 36s (-192m 27s) (10000 13%) 2.2786
44m 28s (-177m 55s) (15000 20%) 1.9853
59m 10s (-162m 44s) (20000 26%) 1.7450
73m 46s (-147m 32s) (25000 33%) 1.5624
88m 33s (-132m 49s) (30000 40%) 1.3690
103m 17s (-118m 3s) (35000 46%) 1.2477
118m 19s (-103m 31s) (40000 53%) 1.0808
133m 23s (-88m 55s) (45000 60%) 0.9605
148m 42s (-74m 21s) (50000 66%) 0.9047
163m 43s (-59m 32s) (55000 73%) 0.8356
178m 36s (-44m 39s) (60000 80%) 0.7251
193m 17s (-29m 44s) (65000 86%) 0.6843
207m 56s (-14m 51s) (70000 93%) 0.6239
222m 31s (-0m 0s) (75000 100%) 0.5655
> nous sommes seules .
= we re alone .
< we re on our own . <EOS>

> je te suis reconnaissant pour ton aide .
= i am grateful for your help .
< i am grateful for your help . <EOS>

> il est encore celibataire .
= he s still single .
< he s still single . <EOS>

> je ne vais pas m impliquer .
= i am not getting involved .
< i m not going to get involved . <EOS>

> il est votre pere .
= he s your father .
< he s your father . <EOS>

> tu n es pas tom .
= you aren t tom .
< you aren t tom . <EOS>

> vous etes un prisonnier .
= you re a prisoner .
< you re a funny . <EOS>

> je suis conscient des risques .
= i m aware of the risks .
< i m aware of the risks . <EOS>

> je me sens plutot fatigue .
= i m feeling sort of tired .
< i m feeling kind of <EOS>

> je suis desolee si je vous ai deranges .
= i m sorry if i disturbed you .
< i m sorry if i disturbed you . <EOS>

> elle est tres fachee .
= she s very upset .
< she is very upset . <EOS>

> je suis ici pour requerir votre aide .
= i m here to ask for your help .
< i m here to ask for your help . <EOS>

> nous n avons pas toujours raison .
= we aren t always right .
< we aren t right . <EOS>

> j ai une dette envers vous .
= i m indebted to you .
< i m indebted at you . <EOS>

> je suis pret a tout faire pour toi .
= i am ready to do anything for you .
< i am ready to do anything for you . <EOS>

> elles se sourient .
= they re smiling at each other .
< they re smiling at you . <EOS>

> elles sont toutes fausses .
= they re all fake .
< they re all fake . <EOS>

> je ne suis pas toujours libre le dimanche .
= i m not always free on sundays .
< i m not always free on sundays . <EOS>

> perdre me fatigue .
= i m getting tired of losing .
< i m tired . <EOS>

> tu mets ma patience a l epreuve .
= you re testing my patience .
< you re making my patience . <EOS>

> tu vas aussi me manquer .
= i m going to miss you too .
< i m going to miss you a lot . <EOS>

> je me trouve juste derriere vous .
= i m right behind you .
< i m right behind you . <EOS>

> nous entrons en premier .
= we re going in first .
< we re going . . <EOS>

> je ne suis pas maigrichonne .
= i m not skinny .
< i m not skinny . <EOS>

> je suis certain d avoir ferme le gaz .
= i m sure i turned off the gas .
< i m sure i turned off the gas . <EOS>

> je n en suis pas fiere .
= i m not proud of that .
< i m not proud of it . <EOS>

> il n est pas en ville .
= he s out of town .
< he s out of town . <EOS>

> c est une personne adorable .
= he is a lovable person .
< he is a lovely person . <EOS>

> je suis lave .
= i m cleaned out .
< i m out . <EOS>

> vous etes courtois .
= you re courteous .
< you re courteous . <EOS>


F:\Desktop\NLP>
"""


if __name__ == '__main__':
    pass