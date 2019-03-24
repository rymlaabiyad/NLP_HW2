import pandas as pd
import numpy as np
import nltk
import spacy 


def retrieveData(path):
    '''Retrieves each line of the data file, and splits it into 5 elements'''
    with open(path) as f:
        return [l.strip().split("\t", 5) for l in f]


def process(lines):
    '''lines is an array containing the lines of our data. 
       Each line contains :
        - polarity
        - aspect
        - word whose polarity we evaluate
        - offset
        - sentence
        '''
    data = pd.DataFrame(lines, columns = ['polarity', 'aspect', 'term', 'offsets', 'sentence'])
    
    target = pd.get_dummies(data['polarity'])
    
    # the function below create a new column 'sentiment_terms that :
    # - extract from sentences adj, verb and adverb
    # - without punctuation nor stop words
    # - and lemmatized
    create_sentiment_terms(data, column_name='sentence')
    
    #we extract from the column 'sentiment_terms', the terms in a window of lenght = window_size 
    window_size=4
    data['words_in_window'] = [ extract_window_words(row['sentiment_terms'],row['term'],window_size ) for i,row in data.iterrows()  ]
    
    #the function below creates an average vector of word2vec from words in the column indicated
    #data['avg_word2vec'] = sent2vec(data, column_name='words_in_window')
    
    features = sent2vec(data, column_name='words_in_window')
    return features, target
    
def tokenize(data):
    '''Count the occurrences of each POS tag in a sentence'''
    tokens = []
    pos = []
    punct = []
        
    for i, row in data.iterrows():
        tmp_pos = {}
        tmp_tokens, tmp_punct = bow(row['sentence'])
            
        for token in nltk.pos_tag(tmp_tokens):
            if token[1] in tmp_pos.keys():
                tmp_pos[token[1]] += 1
            else :
                tmp_pos[token[1]] = 1
            
        tokens.append(tmp_tokens)
        punct.append(tmp_punct)
        pos.append(tmp_pos)
            
    tmp = pd.DataFrame(pos)
    tmp.fillna(value = 0, inplace = True)
        
    punct = pd.DataFrame(punct)

    return tokens, pd.concat([data, punct, tmp], axis=1)
    
def bow(sentence):
    '''Tokenizes a sentence, and counts the # exclamation and question marks'''
    tokens = nltk.tokenize.word_tokenize(sentence)
        
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [t for t in tokens if t not in(stop_words)]
        
    punct = {}
    punct['exclamation'] = 0
    punct['question'] = 0
    for t in tokens:
        if t == '!': 
            punct['exclamation'] += 1
        elif t == '?':
            punct['question'] += 1
    tokens = [t for t in tokens if t.isalpha()]    
    #We get rid of capital letters, in order to count word occurence properly
    tokens = [t.lower() for t in tokens]
        
    return tokens, punct


def create_sentiment_terms (data, column_name) :
    ##function that extract from sentences adj, verb and adverb, without punctuation nor stop words, and lemmatized
    nlp = spacy.load('en_core_web_sm')
    sentiment_terms = []
    for sentence in nlp.pipe(data[column_name]):
        if sentence.is_parsed:
            #sentiment_terms.append(' '.join([token.lemma_ for token in sentence if ( not token.is_stop and not token.is_punct and token.pos_ in ["ADJ", "VERB","ADV",'NOUN'] )]) )
            tag_list = ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']
            #sentiment_terms.append(' '.join([token.lemma_ for token in sentence if ( not token.is_stop and not token.is_punct and token.tag_ in tag_list )]) )
            sentiment_terms.append(' '.join([token.lemma_ for token in sentence if ( not token.is_stop and not token.is_punct )]) )
        else:
            sentiment_terms.append('')  
    data['sentiment_terms'] = sentiment_terms

def sent2vec(data, column_name):
    ''' Take in a dataframe, and return the list of vector average of the sentences
    using 'en_core_web_sm' from spacy '''
    np.random.seed(1)
    #Loading 'en_core_web_sm' from spacy
    nlp = spacy.load('en_core_web_sm')
    
    #Initializing output
    avg_word2vec = []
    
    #Creting the vector average for each sentence
    embedding_size = len( nlp('be').vector )
    vect_aleatoire = np.random.rand(embedding_size)*10
    for index, row in data.iterrows():
        vector = np.zeros(embedding_size) #initializing the sum of vector
        length = 0 # number of words in the sentence
        # Getting the vector for each word in the sentence and adding them together
        for word in row[column_name]:
                vector += nlp(word).vector
                length +=1
        if length :       
            vector_average = vector / length # Dividing the sum of vectors to obtain the average
        else :
            vector_average = vect_aleatoire
        avg_word2vec.append(vector_average) 
    return np.array(avg_word2vec)
    
def extract_window_words(sentence, aspect_terms, window_Size ) :
    """ This method takes as input a sentence, and returns list of tuples :
        - the first element is the center word
        - the second is a list of context words
    """
    sentences_list = sentence.split(' ')
    L = len(sentences_list)
    aspect_context = []
    for index, word in enumerate(sentences_list):
        inf = index - window_Size
        sup = index + window_Size + 1
        
        if word in aspect_terms :
            context = [sentences_list[i] for i in range(inf, sup) if 0 <= i < L and i != index and sentences_list[i] not in aspect_context]
            for c in context :
                aspect_context.append(c)
    return aspect_context



path= "/Users/alimrabeth/Desktop/Master Data Sciences & Business Analytics/Data Sciences Elective courses/NLP/Projet 2/exercise2/data/traindata.csv"
lines = retrieveData(path)

x_train,y_train = process(lines)

dev_path = "/Users/alimrabeth/Desktop/Master Data Sciences & Business Analytics/Data Sciences Elective courses/NLP/Projet 2/exercise2/data/devdata.csv"
x_dev, y_dev = process(retrieveData(dev_path))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D

shape = x_train.shape[1]

nn_model = Sequential()
nn_model.add(Dense(128, input_shape=(shape,), activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(3, activation='softmax'))
nn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

nn_model.fit(x= x_train, y=y_train ,epochs=50, validation_data =(x_dev, y_dev))





