import pandas as pd
import numpy as np
import nltk
import spacy as sc
nlp = sc.load('en')


def retrieveData(path):
        '''Retrieves each line of the data file, and splits it into 5 elements'''
        with open(path) as f:
            return [l.strip().split("\t", 5) for l in f]
        sentences.append([str(t) for t in tokenizer(l) if t.is_punct == False and (str(t) == '\n') == False and t.is_digit == False])

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
        
        polarity = pd.get_dummies(data['polarity'], drop_first=True)
        aspect = pd.get_dummies(data['aspect'], drop_first=True)
        data = pd.concat([data, polarity, aspect], axis=1)
        
        tokens, data = tokenize(data)
        
        #TODO word2vec, then add ONLY necessary features to 'data'
        
        # the function below create a new column 'sentiment_terms that :
        # - extract from sentences adj, verb and adverb
        # - without punctuation nor stop words
        # - and lemmatized
        sentiment_terms(data)
        
        return data
    
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
    
def ngram(sentences, n = 2):
    '''Studies the most common combination of words'''
    pass




def sentiment_terms (data) :
    ##function that extract from sentences adj, verb and adverb, without punctuation nor stop words, and lemmatized
    nlp = sc.load('en')
    sentiment_terms = []
    for sentence in nlp.pipe(data['sentence']):
        if sentence.is_parsed:
            sentiment_terms.append(' '.join([token.lemma_ for token in sentence if ( not token.is_stop and not token.is_punct and token.pos_ in ["ADJ", "VERB","ADV"] )]) )
        else:
            sentiment_terms.append('')  
    data['sentiment_terms'] = sentiment_terms


