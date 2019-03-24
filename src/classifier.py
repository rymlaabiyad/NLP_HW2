import preprocessing as proc
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer
import pandas as pd 

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
        data, y_train = proc.process(lines)
        
        vocab=[]
        for sent in data['words_in_window']:
            for w in sent : 
                if w not in vocab:
                    vocab.append(w)
        vocab_size=len(vocab)
        
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.tokenizer.fit_on_texts(data.words_in_window)
        
        sentiment_tokenized = pd.DataFrame(self.tokenizer.texts_to_matrix(data.words_in_window))
        
        self.clf_tok = Sequential()
        self.clf_tok.add(Dense(128, input_shape=(vocab_size,), activation='softmax'))
        self.clf_tok.add(Dense(64, input_shape=(vocab_size,), activation='relu'))
        self.clf_tok.add(Dense(3, activation='softmax'))
        self.clf_tok.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.clf_tok.fit(sentiment_tokenized, y_train, epochs=10, batch_size=32)
        


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        lines = self.retrieveData(datafile)
        data_eval, y_eval = proc.process(lines)
        
        x_eval = pd.DataFrame(self.tokenizer.texts_to_matrix(data_eval.words_in_window))

        dic = {0:'negative', 1:'neutral', 2:'positive'}
        pred = [dic.get(n,n) for n in np.argmax(self.clf_tok.predict(x_eval), 1)]

        return pred
        


