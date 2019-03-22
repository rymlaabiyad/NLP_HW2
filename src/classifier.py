import preprocessing as proc
from sklearn.model_selection import GridSearchCV

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
        tokens, data = proc.process(lines)
        
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svc = svm.SVC(gamma="scale")
        self.clf = GridSearchCV(svc, parameters, cv=5)
        self.clf.fit(data.iloc[:,1:data.shape[1]], data.iloc[:,0])
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        lines = self.retrieveData(datafile)
        tokens, data = proc.process(lines)
        self.clf.predict(data.iloc[:,1:data.shape[1]])



