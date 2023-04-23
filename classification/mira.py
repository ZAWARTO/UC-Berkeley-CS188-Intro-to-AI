# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Mira implementation

import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        newWeights = self.weights.copy()
        bestWeights = newWeights
        bestAccuracy = 0.0

        for c in Cgrid:
            self.initializeWeightsToZero()

            self.weights = newWeights.copy()
            for iteration in range(self.max_iterations):
                print ("Starting iteration ", iteration, "...")
                for i in range(len(trainingData)):
                    f = trainingData[i]
                    y = trainingLabels[i]
                    score = util.Counter()
                    
                    for label in self.legalLabels:
                        score[label] = self.weights[label] * f
                        
                    y_prim = score.argMax()

                    if y_prim != y:
                        tau = min(c, ((self.weights[y_prim] - self.weights[y]) * f + 1.0) / (2*sum(v**2 for v in f.values())))

                        for key in trainingData[i]:
                            self.weights[y][key] += f[key] * tau
                        for key in trainingData[i]:
                            self.weights[y_prim][key] -= f[key] * tau

            guesses = self.classify(validationData)

            count = 0
            for i in range(len(guesses)):
                if guesses[i] == validationLabels[i]:
                    count += 1

            accuracy = float(count)/float(len(guesses))
            
            if accuracy > bestAccuracy:
                bestWeights = self.weights.copy()
                bestAccuracy = accuracy

        self.weights = bestWeights   
                 
    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
       
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = self.weights[label] * datum
            guesses.append(vectors.argMax())
        return guesses