import preprocessing as proc
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D
from sklearn import svm

class Classifier:
    """The Classifier"""
    
    def retrieveData(self, path):
        '''Retrieves each line of the data file, and splits it into 5 elements'''
        with open(path) as f:
            return [l.strip().split("\t", 5) for l in f]

    
    

    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        lines = self.retrieveData(trainfile)
        x_train, y_train = proc.process(lines)
        
        shape = x_train.shape[1]

        self.clf = Sequential()
        self.clf.add(Dense(128, input_shape=(shape,), activation='relu'))
        self.clf.add(Dense(3, activation='softmax'))
        self.clf.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.clf.fit(x=x_train, y=y_train, epochs=50)


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        lines = self.retrieveData(datafile)
        x_eval, y_eval = proc.process(lines)

        pred = []
        for p in self.clf.predict(x_eval):
            i = np.argmax(p)
            if i == 0:
                pred.append('negative')
            elif i == 1:
                pred.append('neutral')
            else:
                pred.append('positive')
        return pred
        


