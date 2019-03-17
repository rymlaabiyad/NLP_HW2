import preprocessing as proc

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
        print(data.shape)
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """




