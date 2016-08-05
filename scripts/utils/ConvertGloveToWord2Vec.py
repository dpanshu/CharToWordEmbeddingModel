import sys
import json

class Convertor(object):

    def __init__(self, inputVectorFileName, outputVectorFileName):
        self.inputVectorFileName = inputVectorFileName
        self.outputVectorFileName =  outputVectorFileName
        self.wordVectors = dict() 

    def ConvertToWord2Vec(self):
        with open(self.inputVectorFileName) as file:
            for line in file:
                tempStorage = line.split() # Split the words and their vector values
                if isinstance(tempStorage[0], unicode):
                    tempWord = tempStorage[0].encode('utf-8')
                else:
                    tempWord = tempStorage[0]
                tempVectors = map(float, tempStorage[1:])
                yield {tempWord:tempVectors} # Returns a generator

    def Beautify(self):
        print "--Converting GLOVE to Word2Vec--"
        tempListWordVectors = self.ConvertToWord2Vec()
        tempWordVectors  = {}
        for item in tempListWordVectors: # Convert a generator to dictionary
            tempWordVectors.update(item)
        return tempWordVectors

    def getWordVectors(self):
        self.wordVectors = self.Beautify()

    def writeToJsonFile(self):
        print "--Writing the wordvectors--"
        with open(self.outputVectorFileName, 'w') as file:
            json.dump(self.wordVectors, file)

def main():
    inputVectorFileName = sys.argv[1]
    outputVectorFileName = sys.argv[2]
    newConvertorElement = Convertor(inputVectorFileName, outputVectorFileName)
    newConvertorElement.getWordVectors()
    newConvertorElement.writeToJsonFile()



if __name__ == "__main__":
    main()