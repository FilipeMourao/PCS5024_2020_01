#neste arquivo estão descritas todas as funções que serão utilizadas para auxiliar na produção do notebook para analise dos dados
import pandas as pd 
import sklearn 
import functools
import xlrd
def getDataframeInSpecificFormat(datasetPath):
    availabelFormats = ['csv','xslx','txt']
    datasetDataFrame = None
    if (availabelFormats[0] in datasetPath):
        datasetDataFrame = pd.read_csv(datasetPath)

    elif (availabelFormats[1] in datasetPath):
        datasetDataFrame = pd.read_excel(datasetPath)
    
    elif (availabelFormats[2] in datasetPath):
        datasetDataFrame = pd.read_csv(datasetPath)
    
    else:
        print('Formato: ' + format + ' desconhedio, formatos disponiveis:')
        stringOfFormats = functools.reduce(lambda element1,element2: element1 + ' ' + element2,availabelFormats)
        print(stringOfFormats)
    return datasetDataFrame
def saveDescriptionOfEachColumn(datasetPath,pathToSaveTable,tableName):
    datasetDataFrame = getDataframeInSpecificFormat(datasetPath)
    datasetColumns = datasetDataFrame.columns
    describedDataFrame = pd.DataFrame(columns = None)
    # describedDataFrame = pd.Dataframe(columns = datasetColumns)
    for column in datasetColumns:
        describedDataFrame[column] = datasetDataFrame[column].describe()
    describedDataFrame.to_excel(pathToSaveTable+tableName)

def getOccurencesOfMissingValuesByColumns(datasetDataFrame):
    dicOfMissingValues = {}
    sizeOfData = len(datasetDataFrame.index)
    print(datasetDataFrame.head())
    for column in datasetDataFrame.columns:
        dicOfMissingValues[column] = list(datasetDataFrame[column]).count("?")*100/sizeOfData
    return(dicOfMissingValues)

def removeColumnsWithMissingValues(datasetDataFrame):
    dicOfMissingValues = getOccurencesOfMissingValuesByColumns(datasetDataFrame)
    columnsToRemove = [key for key,val  in dicOfMissingValues.items() if val > 0 ]
    datasetDataFrameWithoutColumnsWithMissingValues = datasetDataFrame.drop(columns = columnsToRemove, inplace = True)
    print(dicOfMissingValues)
    print(columnsToRemove)
    return datasetDataFrameWithoutColumnsWithMissingValues

def printOccurencesAndPercentagesOfMissingValuesByColumns(datasetPath):    
    datasetDataFrame = getDataframeInSpecificFormat(datasetPath)
    dicOfMissingValues = getOccurencesOfMissingValuesByColumns(datasetDataFrame)
    print("Porcentagem de '?' em cada coluna ")
    for key,val in  dicOfMissingValues.items(): 
        print('{key} - {val:.2f}'.format(key = key, val = val))
