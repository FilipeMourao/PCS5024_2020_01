#neste arquivo estão descritas todas as funções que serão utilizadas para auxiliar na produção do notebook para analise dos dados
import pandas as pd 
import sklearn 
import functools
import xlrd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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
    for column in datasetColumns:
        describedDataFrame[column] = datasetDataFrame[column].describe(include='all')
    describedDataFrame.to_excel(pathToSaveTable+tableName)

def removeColumnsWithMissingValues(datasetDataFrame):
    dicOfMissingValues = getOccurencesOfMissingValuesByColumns(datasetDataFrame)
    columnsToRemove = [key for key,val  in dicOfMissingValues.items() if val > 0 ]
    datasetDataFrameWithoutColumnsWithMissingValues = datasetDataFrame.drop(columns = columnsToRemove)
    return datasetDataFrameWithoutColumnsWithMissingValues

def removeLinesWithMissingValues(datasetDataFrame):
    linesToBeRemoved = []
    for index, row in datasetDataFrame.iterrows():
        if '?' in list(row):
            linesToBeRemoved.append(index)
    datasetDataFrameWithMissingLines = datasetDataFrame.drop(linesToBeRemoved)
    return datasetDataFrameWithMissingLines

def getCorrelationInDataframe(datasetDataFrame,outputName):
    dummiesDataFrame = pd.get_dummies(datasetDataFrame)
    cor = dummiesDataFrame.corr()
    cor_target = abs(cor[outputName]).sort_values(ascending=False)
    return cor_target[1:]

def getCorrelatedDataFrame(datasetDataFrame,outputName,numberOfRelevantVariables):
    cor_target =  getCorrelationInDataframe(datasetDataFrame,outputName)
    columnsNames = list(cor_target.keys()[numberOfRelevantVariables:])
    dummiesDataFrame = pd.get_dummies(datasetDataFrame)
    return dummiesDataFrame.drop(columns = columnsNames)

def prepareDatasetforTraining(datasetDataFrame,outputName,numberOfRelevantVariables):
    dataFrame  = getCorrelatedDataFrame(datasetDataFrame,outputName,numberOfRelevantVariables)
    columns = list(dataFrame.columns)
    columns.remove(outputName)
    columns.append(outputName)
    dataFrame = dataFrame[columns]
    data = dataFrame.to_numpy()  
    dataScaled = preprocessing.MinMaxScaler().fit_transform(data)
    df = pd.DataFrame(dataScaled)
    df.columns = columns
    X_intermediary, X_test, y_intermediary, y_test = train_test_split(df.drop(columns = outputName).to_numpy() ,df[outputName].to_numpy() , test_size=0.2, random_state=42)
    X_train, X_crossVal, y_train, y_crossVal = train_test_split(X_intermediary ,y_intermediary , test_size=0.25, random_state=42)
    return X_train,X_crossVal, X_test,y_train,y_crossVal, y_test
def prepareDatasetforTrainingComparationWithKaggle(datasetDataFrame,outputName,numberOfRelevantVariables):
    #necessario para comparar com o resultado de 88% obtido por um competidor do kaggle
    dataFrame  = getCorrelatedDataFrame(datasetDataFrame,outputName,numberOfRelevantVariables)
    columns = list(dataFrame.columns)
    columns.remove(outputName)
    columns.append(outputName)
    dataFrame = dataFrame[columns]
    data = dataFrame.to_numpy()  
    dataScaled = preprocessing.MinMaxScaler().fit_transform(data)
    df = pd.DataFrame(dataScaled)
    df.columns = columns
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = outputName).to_numpy() ,df[outputName].to_numpy() , test_size=0.2, random_state=100)
    return X_train, X_test,y_train, y_test