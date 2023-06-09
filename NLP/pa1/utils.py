import gensim
import jieba
import numpy as np
import opencc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from gensim.models import Word2Vec

def read_txt():
    cc = opencc.OpenCC('t2s')
    labels = []
    sentences = []
    with open('tmpData//new_corpus.txt', encoding='utf-8') as f:
        for line in f:
            labels.append(line[0])
            # print(label, sentence)
            sentences.append(line[1:])
    with open('tmpData//train.txt', encoding='utf-8') as f:
        for line in f:
            linelist = line.split(',')
            labels.append(linelist[0])
            sentence = cc.convert(''.join(linelist[1:]))
            # print(label, sentence)
            sentences.append(sentence)
    with open('tmpData//test.txt', encoding='utf-8') as f:
        for line in f:
            linelist = line.split(',')
            sentence = cc.convert(''.join(linelist))
            # print(label, sentence)
            sentences.append(sentence)
    return labels, sentences


def is_number(string):
    if string.endswith('%'):
        string = string.replace('%', '')
    try:
        if string == 'NaN':
            return False
        float(string)
        return True
    except ValueError:
        return False

def partition_word(sentences):
    jieba.setLogLevel(jieba.logging.INFO)
    words = []
    for sentence in sentences:
        iterator = jieba.cut(sentence)
        # word = [i[1] for i in enumerate(iterator)]
        # print('word:', word)
        wordline = []
        for item in iterator:
            if item not in ['\n', '，', '、', '：', '(', ')',
                            ' ', '（', '）',
                            '》', '《', '”', '“', '…', '~'] and not is_number(item):
                item = item.replace('！', '!').replace('？', '?')
                wordline.append(item)
        words.append(wordline)
        # words.append([item for item in iterator if not (item =='\n' or item =='，')])
    return words

def delete_stopwords(words, path='tmpData//cn_stopwords.txt'):
    # print("Deleting stopwords...")
    stopwords = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            stopwords.append(str(line)[:-1])
    # words = [item for item in words if item not in stopwords]
    for word_line in words:
        for word in word_line:
            if word in stopwords and word not in ['！', '？']:
                # print(word,' deleted.')
                word_line.remove(word)
    return words

def load_words():
    labels, sentences = read_txt()
    words = partition_word(sentences)
    words = delete_stopwords(words) # 去除停用词后的分词结果
    with open('tmpData//words.txt', 'w', encoding='utf-8') as f:
            # f.write(str(words))
        for wordline in words:
            f.write(' '.join(wordline))
    return words, labels

def get_word_vector(n_dim):
    print("Getting word vectors...")
    sentences = gensim.models.word2vec.Text8Corpus('tmpData//words.txt')
    model = Word2Vec(sentences, vector_size=n_dim, min_count=5, sg=1)
    return model.wv


def get_sentence_encode(words, word_vec, n_dim):
    # 暂且采用平均模式
    print("Computing sentence vector...")
    sentence_vector = []
    for wordline in words:
        res = [0 for i in range(n_dim)]
        cnt = 0
        for word in wordline:
            # print(word, model.wv[word])
            if word in word_vec.index_to_key:
                res += word_vec[word]
                cnt += 1
        if cnt > 0:
            sentence_vector.append([item / cnt for item in res])
        else:
            # print(' '.join([str(item) for item in wordline]))
            sentence_vector.append(res)
    print(len(sentence_vector), len(sentence_vector[0]))
    return np.array(sentence_vector)

def get_sentence_encode_with_tfidf(words, word_vec, n_dim, weight_matrix):
    print("Computing sentence vector with Tfidf Weight Matrix...")
    # np.save("weight_matrix.npy", np.array(weight_matrix))
    with open('tmpData//weight_matrix.txt', 'w') as f:
        for line in weight_matrix:
            f.write(' '.join([str(item) for item in line]))
            f.write('\n')
    sentence_vector = []
    for idx in range(len(words)):
        wordline = words[idx]
        res = [0 for i in range(n_dim)]
        for jdx in range(len(wordline)):
            word = wordline[jdx]
            if word in word_vec.index_to_key:
                res += word_vec[word] * weight_matrix[idx][jdx]
        # print(''.join(wordline), 'val=', sum(res))
        sentence_vector.append(res)
    print(len(sentence_vector), len(sentence_vector[0]))
    return np.array(sentence_vector)




def Normalization(X):
    for col in range(len(X[0])):
        cur = X[:, col]
        # mean_val = np.mean(cur)
        max_val = np.max(cur)
        min_val = np.min(cur)
        for idx in range(len(cur)):
            cur[idx] = (cur[idx] - min_val) / (max_val - min_val)
    return X

def get_dataset(n_dim, method, load=False):
    '''合并train和test的X，以Y的长度作区分'''
    print("Get dataset from [", method, ']')
    if load:
        X = np.load('X.npy').tolist()
        Y = np.load('Y.npy').tolist()
        return X, Y
    words, Y = load_words()
    if method == 'word2vec':
        word_vec = get_word_vector(n_dim)
        X = get_sentence_encode(words, word_vec, n_dim)
        X = Normalization(X)
    elif method == 'CountVectorizer':
        cv = CountVectorizer(max_df=0.9, min_df=3)
        sentences = []
        for wordline in words:
            sentences.append(' '.join(wordline))
        X = cv.fit_transform(sentences).toarray()
    elif method == 'TfidfVectorizer':
        # tf = TfidfVectorizer(max_df=0.7, min_df=5, token_pattern=r"(?u)\b\w+\b|[!?！？]")
        tf = TfidfVectorizer(max_df=0.9, min_df=3, token_pattern=r"(?u)\b\w+\b|[!?！？]")
        # tf = TfidfVectorizer(max_df=0.95, min_df=1, token_pattern=r"(?u)\b\w+\b|[!?！？]")
        sentences = []
        for wordline in words:
            sentences.append(' '.join(wordline))
        X = tf.fit_transform(sentences).toarray()
    elif method == 'word2vec+TfidfVectorizer':
        # tf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[!?！？]")
        tf = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[!?！？]")
        sentences = []
        for wordline in words:
            sentences.append(' '.join(wordline))
        # print(sentences)
        word_vec = get_word_vector(n_dim)
        weight_matrix = tf.fit_transform(sentences).toarray()
        X = get_sentence_encode_with_tfidf(words, word_vec, n_dim, weight_matrix)
        X = Normalization(X)

    X, Y = np.array(X), np.array(Y)
    # np.save('X.npy', X)
    # np.save('Y.npy', Y)
    return X, Y

if __name__ == '__main__':
    # load_words()
    # get_dataset(train=True, n_dim=100) # word2vec

    # print(cv.get_feature_names_out())
    X, Y = get_dataset(n_dim=1000, method='CountVectorizer', load=False)
    print(X.shape)


        