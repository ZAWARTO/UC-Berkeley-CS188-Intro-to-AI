# perceptron_pacman.py
# --------------------



# Perceptron implementation for apprenticeship learning

import DataLoad
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        #print "Classify self.weights:", self.weights
        data = DataLoad.loadValidationDataPacman()
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        #print "guesses:", guesses
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        
        
        
        trainingData = DataLoad.loadTrainingDataPacman()
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        #print "HASIERA: ", self.weights
        
        for iteration in range(self.max_iterations):
            #print "Starting iteration ", iteration, "..."
            
            for i in range( len(trainingData) ):
                scores=[]
                estado = trainingData[i]
                accionTomada = trainingLabels[i]
                #print "estado" , estado
                #print "accionTomada" , accionTomada
                #input("sartu zerbait")
                
                movimientos = estado[0]
                #print "movimientos" , movimientos
                accionesDisponibles = movimientos.keys() #estado[1]                
                print ("movimientos" , movimientos)
                #print "accionesDisponibles" , accionesDisponibles
                # print "accionesDisponibles" , accionesDisponibles
                 
                for accion in accionesDisponibles:
                    print ("movimiento" , movimientos[accion])
                    #movimiento=movimientos[accion]
                    
                    atributo=str(list(movimientos[accion].keys())[0])
                    print ("   atributo" , atributo)
                    valor=movimientos[accion].get('foodCount')
                    
                    print ("   valor" , valor)
                    #atributo , valor = movimientos[accion].items()[0]
                    #atributo , valor = movimientos[accion][0]                        
                    #print "   atributo" , atributo
                    #print "   valor" , valor
                    #print "self.weights[atributo]", self.weights[atributo] 
                    scores.append(self.weights[atributo] * valor)
                #print "ESTADO: ", estado
                print( "SCORES:", scores)
                max=scores[0]
                ind=0
                #print "len(scores):", len(scores)

                #idxLabel = scores.index(max(scores))
                #print "idxLabel:", idxLabel
                for i in [1,len(scores)-1]:
                    #print "iiiiii:", i
                    if max<scores[i]:
                        max=scores[i]
                        ind=i
                #print "MAX:", max
                print ("MAX_IND:", ind)
                print ("posibleActions: ", list(accionesDisponibles))
                #accion=accionesDisponibles[ind]
                accion=list(accionesDisponibles)[ind]
                print ("ACCION: ", accion)
                if accion != accionTomada:
                    #atributo , valor = movimientos[accionTomada].items()[0]
                    atributo = list(movimientos[accionTomada].keys())[0]
                    valor = movimientos[accionTomada].get('foodCount')
                    self.weights[atributo] = self.weights[atributo] + valor
                    
                    #atributo , valor = movimientos[accion].items()[0]
                    atributo = list(movimientos[accion].keys())[0]
                    valor = movimientos[accion].get('foodCount')
                    self.weights[atributo] = self.weights[atributo] - valor
            
                #print "accion: ", accion
                #print "accionTomada: ", accionTomada
                #input("sartu zerbait")
                #print "   weights" , self.weights
                #input("ALGO") 
                
                
                
                
