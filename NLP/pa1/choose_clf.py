import numpy as np
import torch
import LSTM
import utils
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def clf_evals():
    n_dim = 500
    X_raw, Y = utils.get_dataset(n_dim, 'TfidfVectorizer')
    Y = np.array([eval(i) for i in Y])
    X_train, X_test = X_raw[:len(Y)], X_raw[len(Y):]
    print('X_train:',X_train.shape,'X_test:', X_test.shape,'Y', Y.shape)

    train_len = int(0.8 * len(X_train))
    X_train, X_eval = X_train[:train_len], X_train[train_len:]
    Y_train, Y_eval = Y[:train_len], Y[train_len:]

    classifiers = []

    # clf = LSTM.MyLSTM(X_train.shape[-1], hidden_layer_size=300, output_size=1);classifiers.append(clf)
    clf = GradientBoostingClassifier(n_estimators=20);classifiers.append(clf)
    clf = AdaBoostClassifier();classifiers.append(clf)
    clf = tree.DecisionTreeClassifier();classifiers.append(clf)
    clf = MLPClassifier();classifiers.append(clf)
    clf = MLPClassifier(hidden_layer_sizes=1, activation='logistic', random_state=0, max_iter=200);classifiers.append(clf)
    clf = SVC(kernel='rbf', verbose=True);classifiers.append(clf)
    clf = RandomForestClassifier(n_estimators=20);classifiers.append(clf)
    clf = MultinomialNB();classifiers.append(clf)
    clf = LogisticRegression(penalty='l2');classifiers.append(clf)

    print("Training classifiers...")

    idxs = []
    accs = []
    for idx in range(len(classifiers)):
        clf = classifiers[idx]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_eval)
        acc = accuracy_score(Y_pred, Y_eval)
        print('clf %d accuracy=%.4f%%'%(idx, 100*acc))

        idxs.append(idx)
        accs.append(acc)

    plt.ylim((0.5,1))
    plt.plot(idxs, accs)
    plt.show()

def method_eval():
    n_dim = 500
    X_trains, X_evals, Y_trains, Y_evals = [], [], [], []
    methods = ['word2vec', 'CountVectorizer', 'TfidfVectorizer']
    for method in methods:
        X_raw, Y = utils.get_dataset(n_dim, method)
        Y = np.array([eval(i) for i in Y])
        X_train, X_test = X_raw[:len(Y)], X_raw[len(Y):]
        print('X_train:',X_train.shape,'X_test:', X_test.shape,'Y', Y.shape)

        train_len = int(0.8 * len(X_train))
        X_train, X_eval = X_train[:train_len], X_train[train_len:]
        Y_train, Y_eval = Y[:train_len], Y[train_len:]
        X_trains.append(X_train)
        X_evals.append(X_eval)
        Y_trains.append(Y_train)
        Y_evals.append(Y_eval)

    idxs, accs = [], []
    for idx in range(len(X_trains)):
        X_train, X_eval, Y_train, Y_eval = X_trains[idx], X_evals[idx], Y_trains[idx], Y_evals[idx]
        clf = MLPClassifier(hidden_layer_sizes=1, activation='logistic', random_state=0, max_iter=200);
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_eval)
        acc = accuracy_score(Y_pred, Y_eval)
        print('%s accuracy=%.4f%%' % (methods[idx], 100 * acc))

        idxs.append(idx)
        accs.append(acc)

    plt.ylim((0.5,1))
    plt.plot(idxs, accs)
    plt.show()


if __name__ == '__main__':
    clf_evals()
    method_eval()