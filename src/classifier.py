import preprocessing as proc
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
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

        self.nn_model = Sequential()
        self.nn_model.add(Dense(128, input_shape=(shape,), activation='relu'))
        self.nn_model.add(Conv1D(16,8, padding='same',activation='relu'))
        self.nn_model.add(Dense(3, activation='softmax'))
        self.nn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.nn_model.fit(x= x_train, y=y_train ,epochs=100)
        
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        lines = self.retrieveData(datafile)
        x_eval, y_eval = proc.process(lines)
        pred = np.argmax(self.nn_model.predict(x_eval),1)
        ## !!!!!!! pred est une array avec des 0, 1 ou 2. 
        ## Il faudrait reussir a transformer les 0,1 ou 2 en positif, neutre, négatif. 0 ne signifie pas forcément 
        


