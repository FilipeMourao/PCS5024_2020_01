import FuncoesAuxiliares as fa
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
def applyKNNWithDifferentHyperparameters(X_train, X_test, y_train, y_test,numberOfRelevantVariables):
    listOfDicts = []
    numberOfNeighboorsList = [10,20,30,50,60,70,80,90,100]
    for numberOfNeighboorks in numberOfNeighboorsList:
        peformance_dict = {}
        peformance_dict['number of neighbors'] = numberOfNeighboorks
        neigh = KNeighborsClassifier(n_neighbors=numberOfNeighboorks)
        y_predict = neigh.fit(X_train, y_train).predict(X_test)
        metrics = precision_recall_fscore_support(np.array(y_test), y_predict,average = 'binary')
        peformance_dict['number of relevant variables'] = numberOfRelevantVariables
        peformance_dict['accuracy'] = accuracy_score(np.array(y_test), y_predict)
        peformance_dict['precision'] = metrics[0]
        peformance_dict['recall'] = metrics[1]
        peformance_dict['f1-score'] = metrics[2]
        listOfDicts.append(peformance_dict)
    peformanceDataFrame = pd.DataFrame(listOfDicts) 
    return peformanceDataFrame
def applyNaiveBayesWithDifferentHyperparameters(X_train, X_test, y_train, y_test,numberOfRelevantVariables):
    naiveBayesOptimization = ['gaussian','multinominal','bernouli']
    listOfDicts = []
    for naiveBayesOptimizationParameter in naiveBayesOptimization: 
        peformance_dict = {}
        peformance_dict['method optimization'] = naiveBayesOptimizationParameter
        if naiveBayesOptimizationParameter == 'gaussian':
            nbo = GaussianNB()
        elif naiveBayesOptimizationParameter == 'multinominal' : 
            nbo = MultinomialNB()
        else:
            nbo = BernoulliNB()
        y_predict = nbo.fit(X_train, y_train).predict(X_test)
        metrics = precision_recall_fscore_support(np.array(y_test), y_predict,average = 'binary')
        peformance_dict['number of relevant variables'] = numberOfRelevantVariables
        peformance_dict['accuracy'] = accuracy_score(np.array(y_test), y_predict)
        peformance_dict['precision'] = metrics[0]
        peformance_dict['recall'] = metrics[1]
        peformance_dict['f1-score'] = metrics[2]
        listOfDicts.append(peformance_dict)
    peformanceDataFrame = pd.DataFrame(listOfDicts)
    return peformanceDataFrame
def applyLogisticRegressionDifferentHyperparameters(X_train, X_test, y_train, y_test,numberOfRelevantVariables):
    listOfDicts = []
    Cs = [2**i for i in range (-5,5)]
    for C in Cs:
        peformance_dict = {}
        peformance_dict['C'] = C
        y_predict = LogisticRegression(C=C,random_state=0).fit(X_train, y_train).predict(X_test)
        metrics = precision_recall_fscore_support(np.array(y_test), y_predict,average = 'binary')
        peformance_dict['number of relevant variables'] = numberOfRelevantVariables
        peformance_dict['accuracy'] = accuracy_score(np.array(y_test), y_predict)
        peformance_dict['precision'] = metrics[0]
        peformance_dict['recall'] = metrics[1]
        peformance_dict['f1-score'] = metrics[2]
        listOfDicts.append(peformance_dict)
    peformanceDataFrame = pd.DataFrame(listOfDicts) 
    return peformanceDataFrame
def applyDecisionTreeDifferentHyperparameters(X_train, X_test, y_train, y_test,numberOfRelevantVariables):
    listOfDicts = []
    max_depths = [int(x) for x in np.linspace(10, 110, num = 11)]
    for max_depth in max_depths:
        peformance_dict = {}
        peformance_dict['max_depths'] = max_depth
        dTC = tree.DecisionTreeClassifier()
        y_predict = dTC.fit(X_train, y_train).predict(X_test)
        metrics = precision_recall_fscore_support(np.array(y_test), y_predict,average = 'binary')
        peformance_dict['number of relevant variables'] = numberOfRelevantVariables
        peformance_dict['accuracy'] = accuracy_score(np.array(y_test), y_predict)
        peformance_dict['precision'] = metrics[0]
        peformance_dict['recall'] = metrics[1]
        peformance_dict['f1-score'] = metrics[2]
        listOfDicts.append(peformance_dict)
    peformanceDataFrame = pd.DataFrame(listOfDicts) 
    return peformanceDataFrame
def applyRandomForestDifferentHyperparameters(X_train, X_test, y_train, y_test,numberOfRelevantVariables):
    listOfDicts = []
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
    max_depths = [int(x) for x in np.linspace(10, 110, num = 5)]
    for n_estimator in n_estimators:
        for max_depth in max_depths:
            peformance_dict = {}
            peformance_dict['max_depth'] = max_depth
            peformance_dict['n_estimator'] = n_estimator
            rFC = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth, random_state=0)
            y_predict = rFC.fit(X_train, y_train).predict(X_test)
            metrics = precision_recall_fscore_support(np.array(y_test), y_predict,average = 'binary')
            peformance_dict['number of relevant variables'] = numberOfRelevantVariables
            peformance_dict['accuracy'] = accuracy_score(np.array(y_test), y_predict)
            peformance_dict['precision'] = metrics[0]
            peformance_dict['recall'] = metrics[1]
            peformance_dict['f1-score'] = metrics[2]
            listOfDicts.append(peformance_dict)
    peformanceDataFrame = pd.DataFrame(listOfDicts) 
    return peformanceDataFrame
