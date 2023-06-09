import torch
from torch import nn
import numpy as np
import torch
import LSTM
import utils
from gensim.models import Word2Vec
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    n_dim = 500
    max_epochs = 30
    # method = 'word2vec'
    # method = 'CountVectorizer'
    method = 'TfidfVectorizer'
    # method = 'word2vec+TfidfVectorizer'
    X_raw, Y = utils.get_dataset(n_dim, method)
    Y = np.array([eval(i) for i in Y])
    X_train, X_test = X_raw[:len(Y)], X_raw[len(Y):]
    print('X_train:', X_train.shape, 'X_test:', X_test.shape, 'Y:', Y.shape)


    print("Training classifier...")
    # clf = LSTM.MyLSTM(X_train.shape[-1], hidden_layer_size=300, output_size=1)
    # clf = GradientBoostingClassifier(n_estimators=200)
    # clf = AdaBoostClassifier()
    # clf = tree.DecisionTreeClassifier()
    # clf = MLPClassifier() # 0.826
    clf = MLPClassifier(hidden_layer_sizes=1, activation='logistic', random_state=0, max_iter=120)  # 0.874(0.7,5, 200)!!
    # 0.884(0.9, 3, 100) # 0.885(0.9, 3, 125) # 0.887(0.9, 3, 120)
    # clf = SVC(kernel='rbf', verbose=True)
    # clf = RandomForestClassifier(n_estimators=100) # 0.812(cv) 0.83(tf)
    # clf = MultinomialNB() # 0.792
    # clf = LogisticRegression(penalty='l2')

    clf.fit(X_train, Y)
    Y_pred = clf.predict(X_test)

    path = '201300096.txt'
    print("Writing results into", path)
    with open(path, 'w') as f:
        for item in Y_pred:
            f.write(str(item))
            f.write('\n')

