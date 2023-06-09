import os

FIX_POS = os.listdir('corpus//fix_pos')
FIX_NEG = os.listdir('corpus//fix_neg')

sentences, Y = [], []
for file in FIX_POS:
    sentence = []
    with open('corpus//fix_pos//'+file, encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            sentence.append(line)
        # print(sentence)
        sentences.append(' '.join(sentence))
        Y.append(1)

for file in FIX_NEG:
    sentence = []
    with open('corpus//fix_neg//'+file, encoding='utf-8') as f:
        for line in f:
            if not line == '\n':
                line = line.replace('\n', '')
                sentence.append(line)
        sentences.append(' '.join(sentence))
        Y.append(0)

with open('new_corpus.txt', 'w', encoding='utf-8') as f:
    for idx in range(len(Y)):
        f.write(str(Y[idx]))
        f.write(',')
        f.write(sentences[idx])
        f.write('\n')
