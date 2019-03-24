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
    data = pd.DataFrame(lines, columns=['polarity', 'aspect', 'term', 'offsets', 'sentence'])

    polarity = pd.get_dummies(data['polarity'], drop_first=True)
    aspect = pd.get_dummies(data['aspect'], drop_first=True)
    data = pd.concat([data, polarity, aspect], axis=1)

    data.replace({'polarity': {'positive': 1, 'neutral': 0, 'negative': -1}}, inplace=True)

    #_, data = tokenize(data)
    #data = pd.concat([data, vector_context(data['sentence'])], axis=1)

    data = pd.concat([data, vector_context(sentiment_terms(data))], axis=1)

    target_scalar = data['polarity']
    target_vec = data.iloc[:, 5:7]

    return target_scalar, target_vec, data.iloc[:, 7:len(data.columns)]
    
def tokenize(data):
    '''Count the occurrences of each POS tag in a sentence'''
    tokens = []
    pos = []
    tags = ['NN', 'JJ', 'VB']
    punct = []
        
    for i, row in data.iterrows():
        tmp_tokens, tmp_punct = bow(row['sentence'])

        tmp_pos = {}
        for tag in tags:
            tmp_pos[tag] = 0


        for token in nltk.pos_tag(tmp_tokens):
            if token[1] in tmp_pos.keys():
                tmp_pos[token[1]] += 1
            
        tokens.append(tmp_tokens)
        punct.append(tmp_punct)
        pos.append(tmp_pos)
            
    tmp = pd.DataFrame(pos)
    tmp.fillna(value=0, inplace=True)
        
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
    
def ngram(sentences, n = 2):
    '''Studies the most common combination of words'''
    pass




def sentiment_terms (data) :
    ##function that extract from sentences adj, verb and adverb, without punctuation nor stop words, and lemmatized
    nlp = spacy.load('en')
    sentiment_terms = []
    for sentence in nlp.pipe(data['sentence']):
        if sentence.is_parsed:
            #sentiment_terms.append(' '.join([token.lemma_ for token in sentence if ( not token.is_stop and not token.is_punct and token.pos_ in ["ADJ", "VERB","ADV",'NOUN'] )]) )
            tag_list = ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']
            sentiment_terms.append(' '.join([token.lemma_ for token in sentence if ( not token.is_stop and not token.is_punct and token.tag_ in tag_list )]) )
        else:
            sentiment_terms.append('')  
    return sentiment_terms

def vector_context(column):
    ''' Take in a dataframe, and return the list of vector average of the sentences
    using 'en_core_web_sm' from spacy '''
    
    #Loading 'en_core_web_sm' from spacy
    nlp = spacy.load('en_core_web_sm')
    
    #Initializing output
    avg_word2vec = []
    
    #Creting the vector average for each sentence
    embedding_size = len(nlp(column[0])[0].vector)
    for sentence in column:
        vector = np.zeros(embedding_size) #initializing the sum of vector
        length = 0 # number of words in the sentence
        
        # Getting the vector for each word in the sentence and adding them together
        for word in nlp(sentence):
            vector += word.vector
            length +=1
                
        vector_average = vector / length # Dividing the sum of vectors to obtain the average
        avg_word2vec.append(vector_average)

    df = pd.DataFrame(avg_word2vec)
    return df
    


