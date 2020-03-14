import FuncoesAuxiliares as fa
import pandas as pd
def main():
    datasetPath = '../Kaggle/adult.csv'
    pathToSaveTable = '../Gráficos, tabelas e relatório/'
    tableNameDescribed = 'describedAdult.xlsx'
    correlationImageName = 'correlation.png'
    outputName = 'income'
    numberOfRelevantVariables = 15
    datasetDataFrame = fa.getDataframeInSpecificFormat(datasetPath) 
    datasetDataFrame[outputName] = [0 if val == '<=50K' else 1 for val in  datasetDataFrame[outputName] ]    
    dataFrameWithRemovedColumns = fa.removeColumnsWithMissingValues(datasetDataFrame)
    dataFrameWithRemovedLines = fa.removeLinesWithMissingValues(datasetDataFrame)

    fa.getMostCorrelatedColumns(dataFrameWithRemovedLines,outputName,numberOfRelevantVariables)
    fa.saveDescriptionOfEachColumn(datasetPath,pathToSaveTable,tableNameDescribed)
    fa.saveCorrelationImage(datasetDataFrame,pathToSaveTable,correlationImageName,outputName,numberOfRelevantVariables)

if __name__ == "__main__":
    main()