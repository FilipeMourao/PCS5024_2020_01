import FuncoesAuxiliares as fa
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
def applyKNNWithDifferentHyperparameters(dataFrame, outputPath,fileName,outputName,numberOfRelevantVariables):
    listOfDicts = []
    numberOfNeighboorsList = [10,20,30,50]
    for numberOfVariables in numberOfRelevantVariables:
        X_train, X_test, y_train, y_test = fa.prepareDatasetforTraining(dataFrame,outputName,numberOfVariables)
        for numberOfNeighboorks in numberOfNeighboorsList:
            peformance_dict = {}
            peformance_dict['number of relevant variables'] = numberOfVariables
            peformance_dict['number of neighboors'] = numberOfNeighboorks
            neigh = KNeighborsClassifier(n_neighbors=numberOfNeighboorks)
            neigh.fit(X_train, y_train)
            y_predict = neigh.predict(X_test)
            metrics = precision_recall_fscore_support(np.array(y_test), y_predict,average = 'binary')
            peformance_dict['accuracy'] = accuracy_score(np.array(y_test), y_predict)
            peformance_dict['precision'] = metrics[0]
            peformance_dict['recall'] = metrics[1]
            peformance_dict['f1-score'] = metrics[2]
            listOfDicts.append(peformance_dict)
    peformanceDataFrame = pd.DataFrame(listOfDicts)
    peformanceDataFrame.to_excel(outputPath+fileName,index = False)
def applyNaiveBayesWithDifferentHyperparameters(X_train, X_test, y_train, y_test, outputPath,fileName,numberOfRelevantVariables):
    listOfDicts = []
    peformance_dict = {}
    peformanceDataFrame = pd.DataFrame(listOfDicts)
    peformanceDataFrame.to_excel(outputPath+fileName,index = False)
