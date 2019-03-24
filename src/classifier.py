import preprocessing as proc

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM


class Classifier:
    """The Classifier"""

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        lines = proc.retrieveData(trainfile)
        target_scalar, target_vec, data = proc.process(lines)

        if self.algorithm == 'svm':
            model = SVC(gamma="scale"
                        # , class_weight='balanced'
                        )
            parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100]}
            self.clf = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score), cv=5)
            self.clf.fit(data, target_scalar)

        elif self.algorithm == 'logreg':
            model = LogisticRegression(multi_class='multinomial'
                                       # , class_weight='balanced'
                                       )
            parameters = {"penalty": ['l2', 'l1'], 'solver': ["lbfgs", "sag", "saga"]}
            self.clf = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score), cv=5)
            self.clf.fit(data, target_scalar)

            self.clf = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score), cv=5)
            self.clf.fit(data, target_scalar)

        elif self.algorithm == 'nn':
            self.clf = Sequential()
            self.clf.add(Dense(256, activation='softmax', input_shape=(data.shape[1],)))
            self.clf.add(Dense(128, activation='softmax'))
            self.clf.add(Dropout(0.4))
            self.clf.add(Dense(64, activation='softmax'))
            self.clf.add(Dense(32, activation='softmax'))
            self.clf.add(Dropout(0.4))
            self.clf.add(Dense(16, activation='softmax'))
            self.clf.add(Dense(2, activation='relu'))
            self.clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.clf.fit(data, target_vec, batch_size=64, validation_split=0.3, epochs=20,
                         callbacks=[EarlyStopping(patience=5)])

        elif self.algorithm == 'lstm':
            self.clf = Sequential()
            self.clf.add(Embedding(150, 200, input_length=data.shape[1]))
            self.clf.add(LSTM(100, dropout=0.5, recurrent_dropout=0.2))
            self.clf.add(Dense(2, activation='sigmoid'))
            # self.clf.add(TimeDistributed(Dense(vocabulary)))
            # self.clf.add(Activation('softmax'))
            self.clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.clf.fit(data, target_vec, batch_size=64,
                         validation_split=0.3, epochs=20, callbacks=[EarlyStopping(patience=5)])

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        lines = proc.retrieveData(datafile)
        _, _, data = proc.process(lines)

        pred = []

        scalar = ['svm', 'logreg', 'randomForest']
        if self.algorithm in scalar:
            for p in self.clf.predict(data):
                if p == 1:
                    pred.append('positive')
                elif p == -1:
                    pred.append('negative')
                elif p == 0:
                    pred.append('neutral')
                else:
                    print('PROBLEM')
        else:
            for p in self.clf.predict(data):
                if (p[0] < 0.1) & (p[1] < 0.1):
                    pred.append('negative')
                elif (p[0] > 0.9) & (p[1] < 0.1):
                    pred.append('neutral')
                else:
                    pred.append('positive')
        return pred
