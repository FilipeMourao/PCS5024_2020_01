import FuncoesAuxiliares as fa
import pandas as pd
import FuncoesDeMachineLearning as fm
def main():
    datasetPath = '../Kaggle/adult.csv'
    pathToSaveTable = '../Gráficos, tabelas e relatório/'
    tableNameDescribed = 'describedAdult.xlsx'
    correlationImageName = 'correlation.png'
    outputName = 'income'
    numberOfRelevantVariables = [3,5,8,10,20]

    datasetDataFrame = fa.getDataframeInSpecificFormat(datasetPath) 
    datasetDataFrame[outputName] = [0 if val == '<=50K' else 1 for val in  datasetDataFrame[outputName] ]    
    dataFrameWithRemovedColumns = fa.removeColumnsWithMissingValues(datasetDataFrame)
    dataFrameWithRemovedLines = fa.removeLinesWithMissingValues(datasetDataFrame)
    
    # fa.printOccurencesAndPercentagesOfMissingValuesByColumns(datasetPath)
    
    # fa.saveCorrelationImage(datasetDataFrame,pathToSaveTable,correlationImageName,outputName,numberOfRelevantVariables = 10)
    
    # ## Aplica algoritmo knn com variacao de parametros 
    # fileName = 'KNN_com_linhas_removidas.xlsx'
    # fm.applyKNNWithDifferentHyperparameters(dataFrameWithRemovedLines,pathToSaveTable, fileName,outputName,numberOfRelevantVariables)
    
    # ## Aplica algoritmo naive bayes com variacao de parametros 
    # fileName = 'Naive_Bayes_com_linhas_removidas.xlsx'
    # fm.applyNaiveBayesWithDifferentHyperparameters(dataFrameWithRemovedLines, pathToSaveTable,fileName,outputName,numberOfRelevantVariables)
if __name__ == "__main__":
    main()