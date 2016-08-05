import os
import sys
import json
import numpy as np
from keras.layers import Input, Dense
from keras.layers import Embedding, Flatten, TimeDistributed
from keras.models import Model

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class SimpleDenseLayerArchitecture(object):

    def __init__(self, CharModelPreprocessingElement, outputArchitectureFile, outputWeightsFile):
        self.CharModelPreprocessingElement = CharModelPreprocessingElement
        self.outputArchitectureFile = outputArchitectureFile
        self.outputWeightsFile = outputWeightsFile
        self.model = None

    def Architecture(self):
        inputLayer = Input(shape = (self.CharModelPreprocessingElement.inputShape, ), dtype='int32')
        embeddingLayer = Embedding(output_dim=25, input_dim=55, 
                                   input_length=self.CharModelPreprocessingElement.inputShape)(inputLayer)
        firstDenseLayer = TimeDistributed(Dense(100, activation='relu'))(embeddingLayer)
        secondDenseLayer = TimeDistributed(Dense(1, activation='tanh'))(firstDenseLayer)
        flattenSecondLayer = Flatten()(secondDenseLayer)
        outputLayer = Dense(50, activation='linear')(flattenSecondLayer)
        self.model = Model(input = inputLayer, output = outputLayer)

    def CompileArchitecture(self):
        self.model.compile(optimizer='rmsprop',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def trainModel(self):
        self.model.fit(self.CharModelPreprocessingElement.wordIntSequence, self.CharModelPreprocessingElement.vectors)

    def saveModel(self):
        architecture = self.model.to_json()
        with open(self.outputArchitectureFile,'w') as file:
            json.dump(architecture, file)
        self.model.save_weights(self.outputWeightsFile)

class CharModelPreprocessing(object):

    def __init__(self, inputFile):
        self.inputFile = CUR_DIR + "/../data/" + inputFile
        self.wordIntSequence = list()

    def GetWordsAndVectors(self):
        with open(self.inputFile) as file:
            word2vec = json.load(file)
        self.words = word2vec.keys()
        self.vectors = word2vec.values()

    def ConvertWordToIntSequence(self):
        with open(CUR_DIR+'/../data/CharacterIndex.json') as file:
            wordIntIndex = json.load(file)

        self.wordIntSequence = np.zeros(shape=(len(self.words), 10))
        for index1, word in enumerate(self.words):
            for index2, character in enumerate(word[:10]):
                try:
                    self.wordIntSequence[index1][index2] = wordIntIndex['character']
                except:
                    pass

    def Shapes(self):
        self.inputShape = 10
        self.outputShape = len(self.vectors[0])


def main():

    inputFile = sys.argv[1]
    outputArchitectureFile = sys.argv[2]
    outputWeightsFile = sys.argv[3]

    # Preprocess the word vectors 
    CharModelPreprocessingElement = CharModelPreprocessing(inputFile)

    # Loading Wordvectors
    print "--Loading WordVectors--"
    CharModelPreprocessingElement.GetWordsAndVectors()

    print "--Converting words to integer sequence--"
    CharModelPreprocessingElement.ConvertWordToIntSequence()
    CharModelPreprocessingElement.Shapes()

    print "--Initiating Keras model--"
    SimpleDenseLayerArchitectureElement = SimpleDenseLayerArchitecture(CharModelPreprocessingElement, 
                                                                       outputArchitectureFile, outputWeightsFile)
    SimpleDenseLayerArchitectureElement.Architecture()
    SimpleDenseLayerArchitectureElement.CompileArchitecture()

    print "--Training Keras Model--"
    SimpleDenseLayerArchitectureElement.trainModel()

    print "--Saving Model--"
    SimpleDenseLayerArchitectureElement.saveModel()

if __name__=="__main__":
    main()
