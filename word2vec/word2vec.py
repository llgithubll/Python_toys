"""
Created on Sun Apr 22 14:08:02 2018
@author: lilin
"""
import sys
import math
import numpy as np


def load_corpus(filename):
    """Each line in file, convert to token list,
    all list append in ``corpus`` list and return.
    """
    corpus = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            corpus.append(tokens)
        return corpus


def sigmoid(z):
    """Some modify for faster.
    """
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, W1, filename):
    """Save word and it's vector in text file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for w, vec in zip(vocab, W1):
            word = w.word
            vector_str = ','.join([str(val) for val in vec])
            f.write('%s\t%s\n' % (word, vector_str))


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    """Contain words, each word has word string and count in corpus.
    """
    def __init__(self, corpus, min_count):
        self.token2id = {}
        self.id2word = []
        self.min_count = min_count
        i = 0
        for tokens in corpus:
            for token in tokens:
                if token not in self.token2id:
                    self.token2id[token] = len(self.id2word)
                    self.id2word.append(Word(token))
                self.id2word[self.token2id[token]].count += 1
                i += 1
                if i%10000 == 0:
                    sys.stdout.write('\rBuilding vocabulary: %d'
                                     % (len(self.id2word)))
                    sys.stdout.flush()
        self._remove_rare_word()
        print('\nVocabulary size: %d' % (len(self.id2word)))

    def _remove_rare_word(self):
        new_id2word = []
        for word in self.id2word:
            if word.count >= self.min_count:
                new_id2word.append(word)
        #        new_id2word.sort(key=lambda word: word.count, reverse=True)
        new_token2id = {}
        for i, word in enumerate(new_id2word):
            new_token2id[word.word] = i
        self.id2word = new_id2word
        self.token2id = new_token2id

    def __getitem__(self, i):
        return self.id2word[i]

    def __len__(self):
        return len(self.id2word)

    def __iter__(self):
        return iter(self.id2word)

    def __contains__(self, token):
        return token in self.token2id

    def indices(self, tokens):
        return [self.token2id[token] if token in self else -1
                for token in tokens]


class NegativeSampleTable:
    def __init__(self, vocab):
        power = 0.75
        denom = sum([math.pow(w.count, power) for w in vocab])
        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.uint32)
        p = 0
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power)) / denom
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(0, len(self.table), count)
        return [self.table[i] for i in indices]


def train_model(corpus_file,
                window=5,
                dimension=100,
                min_count=5,
                k_negative_sampling=5,
                subsampling=0.001,
                word2vec_file='word2vec.txt'):
    corpus = load_corpus(corpus_file)
    vocab = Vocabulary(corpus, min_count)
    table = NegativeSampleTable(vocab)
    print('Traning on:                %s' % corpus_file)
    print('Context window sizes:      %d' % window)
    print('Word vector dimension:     %d' % dimension)
    print('Truncate min count:        %d' % min_count)
    print('Negative sampling number:  %d' % k_negative_sampling)
    print('Subsampling parameter:     %f' % subsampling)
    print('Save word2vec in:          %s' % word2vec_file)
    W1 = np.random.uniform(-0.5/dimension, 0.5/dimension,
                           size=(len(vocab), dimension))
    W2 = np.zeros(shape=(len(vocab), dimension))
    alpha = 0.025
    total_lines = len(corpus)
    line_cnt = 0
    total_training_words = 0
    for tokens in corpus:
        total_training_words += len(tokens)
    for tokens in corpus:
        line_cnt += 1
        sys.stdout.write('\rProcess in : %d/%d line' % (line_cnt, total_lines))
        sys.stdout.flush()
        line_words_cnt = len(tokens)
        tokens_id = vocab.indices(tokens)
        for i, token_id in enumerate(tokens_id):
            if token_id == -1:
                continue
            # subsampling
            if subsampling > 0:
                # (sqrt(word_freq/0.001)+1) * (0.001/word_freq)
                word_freq = vocab[token_id].count / total_training_words
                keep_probability = (math.sqrt(word_freq / subsampling) + 1) * \
                                   (subsampling / word_freq)
                if keep_probability < np.random.random():
                    continue
            random_window = np.random.randint(low=1, high=window+1)
            context_begin = max(i-random_window, 0)
            context_end = min(i+1+random_window, len(tokens_id))
            context_ids = tokens_id[context_begin:i] + tokens_id[i+1:context_end]
            for context_id in context_ids:
                if context_id == -1:
                    continue
                e = np.zeros(dimension)
                samples = [(token_id, 1)] + [(neg_token_id, 0)
                                             for neg_token_id in table.sample(k_negative_sampling) if neg_token_id != token_id]
                for word_id, label in samples:
                    q = sigmoid(np.dot(W1[context_id], W2[word_id]))
                    g = alpha * (label - q)
                    e += g * W2[word_id]
                    W2[word_id] += g * W1[context_id]
                W1[context_id] += e
            if i % 10000 == 0:
                sys.stdout.write('\rProcess in : %d/%d line, %d/%d word'
                                 % (line_cnt, total_lines, i, line_words_cnt))
                sys.stdout.flush()
    save(vocab, W1, word2vec_file)


def distance(word2vec_file='word2vec.txt'):
    word2vec = {}
    with open(word2vec_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            word = line[0]
            vec_str = line[1].split(',')
            vec = np.array([float(val) for val in vec_str])
            word2vec[word] = vec
    while True:
        ques = input('Enter word or sentence (EXIT to break):')
        if ques == 'EXIT':
            break
        elif ques not in word2vec:
            print('Out of dictionary word!')
        else:
            import heapq
            cloest10 = []
            ques_vec = word2vec[ques]
            for i, word in enumerate(word2vec):
                cos = np.dot(ques_vec, word2vec[word]) / \
                      (np.linalg.norm(ques_vec) * np.linalg.norm(word2vec[word]))
                similarity = 0.5 + 0.5 * cos
                if i < 10:
                    heapq.heappush(cloest10, (similarity, word))
                else:
                    heapq.heappushpop(cloest10, (similarity, word))
            print('similarity  word')
            for similarity, word in heapq.nlargest(10, cloest10):
                print('%10.3f  %s' % (similarity, word))


if __name__ == '__main__':
    import time
    time_begin = time.time()
    train_model('text8')
    print('\nTraining done, total {0} seconds'.format(time.time()-time_begin))
    distance()
