#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aaryanjha
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
import scipy.spatial 
import timeit
from sklearn import model_selection
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
# -----------------------------------------------------------------------------------------

# GIVEN: For use in all testing for the purpose of grading

def testMain():
    pass
    print("========== testAlwaysOneClassifier ==========")
    testAlwaysOneClassifier()
    print("========== testFindNearest() ==========")
    testFindNearest()
    print("========== testOneNNClassifier() ==========")
    testOneNNClassifier()
    print("========== testCVManual(OneNNClassifier(), 5) ==========")
    testCVManual(OneNNClassifier(), 5)
    print("========== testCVBuiltIn(OneNNClassifier(), 5) ==========")
    testCVBuiltIn(OneNNClassifier(), 5)
    print("========== compareFolds() ==========")
    compareFolds()
    print("========== testStandardize() ==========")
    testStandardize()
    print("========== testNormalize() ==========")
    testNormalize()
    print("========== comparePreprocessing() ==========")
    comparePreprocessing()
    print("========== visualization() ==========")
    visualization()
    print("========== testKNN() ==========")
    testKNN()
    print("========== paramSearchPlot() ==========")
    paramSearchPlot()
    print("========== paramSearchPlotBuiltIn() ==========")
    paramSearchPlotBuiltIn()

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------

# GIVEN: For use starting in the "Reading in the data" step

def readData(numRows=None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids",
                 "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("/Users/aaryanjha/Desktop/hw05/data/wine.data", header=None, names=colNames, nrows=numRows)
    # Need to mix this up before doing CV
    wineDF = wineDF.sample(frac=1, random_state=50).reset_index(drop=True)
    return wineDF, inputCols, outputCol

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------

# GIVEN: For use starting in the "Testing AlwaysOneClassifier" step

def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0
    return accuracy

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------


def operationsOnDataFrames():
    d = {'x': pd.Series([1, 2], index=['a', 'b']),
         'y': pd.Series([10, 11], index=['a', 'b']),
         'z': pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')
    cols = ['x', 'z']
    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')
    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------


def testStandardize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    standardize(df, colsToStandardize)
    print("After standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, colsToStandardize].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, colsToStandardize].std(), sep='\n', end='\n\n')

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------


def testNormalize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    normalize(df, colsToStandardize)
    print("After normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, colsToStandardize].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, colsToStandardize].min(), sep='\n', end='\n\n')

# -----------------------------------------------------------------------------------------

class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    def __innit__(self):
        pass
    
    def fit(self, inputDF, outputSeries):
        return self

    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return 1
        else:
            howMany = testInput.shape[0]
            return pd.Series(np.ones(howMany), index=testInput.index, dtype="int64")

def testAlwaysOneClassifier():
    df, inputCols, outputCol = readData()
    testInputDF = df.iloc[:10,1:]
    testOutputSeries = df.iloc[:10,0]
    trainInputDF = df.iloc[10:,1:]
    trainOutputSeries = df.iloc[10:,0]
    
    print("testInputDF:", testInputDF, sep='\n', end='\n\n') 
    print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    print("trainInputDF:", trainInputDF, sep='\n', end='\n\n') 
    print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')

    instance = AlwaysOneClassifier()
    instance = instance.fit(trainInputDF, trainOutputSeries)

    print("---------- Test one example")
    print("Correct answer:", testOutputSeries.iloc[0])
    print("Predicted answer:", instance.predict(testInputDF.iloc[0,:]))
    print("---------- Test the entire test set")
    print("Correct answer:","\n", testOutputSeries.iloc[:])
    print("Predicted answer:","\n", instance.predict(testInputDF))
    print("Accuracy: ", accuracyOfActualVsPredicted(testOutputSeries.iloc[:], instance.predict(testInputDF)))

def findNearestLoop(df, testRow):
    minDis = scipy.spatial.distance.euclidean(df.iloc[0,:], testRow)
    mini= 0
    
    for i in range(df.shape[0]):
        eucliDis = scipy.spatial.distance.euclidean(df.iloc[i,:], testRow)
        if (minDis > eucliDis):
            minDis = eucliDis
            mini = i
    return df.index[mini]

def findNearestHOF(df, testRow):
    distance = df.apply(lambda a: scipy.spatial.distance.euclidean(a, testRow), axis = 1)
    return distance.idxmin()

def testFindNearest():
    df,inputCols, outputCol = readData()
    startTime = timeit.default_timer()
    for i in range (100) :
        findNearestLoop(df.iloc[100:107,:], df.iloc[90,:])
        
    print ("findNearestLoop:", timeit.default_timer() - startTime)
    print(  findNearestLoop(df.iloc[100:107,:], df.iloc[90,:]))

    startTime1 = timeit.default_timer()

    for i in range(100) :
        findNearestHOF(df.iloc[100:107, :], df.iloc[90,:])
        
    print ("findNearestHOF:", timeit.default_timer() - startTime1)
    print(findNearestHOF(df.iloc[100:107, :], df.iloc[90,:]))
    
class OneNNClassifier(BaseEstimator, ClassifierMixin):    
    def __innit__(self):
        inputDF=None
        outputSeries=None

    def fit(self, inputDF, outputSeries):
        self.inputDF=inputDF
        self.outputSeries=outputSeries
        
    def _predictOne(self,testInput):
        return self.outputSeries.loc[findNearestHOF(self.inputDF, testInput)]
    
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return (self._predictOne(testInput))
        else:
            series=testInput.apply(lambda row: self._predictOne(row), axis=1)
            return series
        
def testOneNNClassifier():
     df, inputCols,outputCols= readData()
     print(df)
     test_Input=df.iloc[:10,1:]
     test_Output=df.iloc[:10,0]
     train_Input=df.iloc[10:,1:] 
     train_Output= df.iloc[10:,0]
     test=OneNNClassifier()
     test.fit(train_Input, train_Output)
     print("---------------Test the second row")
     print("Correct Answer: ", test_Output.iloc[2] )
     print("Predicted answer: ",test.predict(test_Input.iloc[2,:]) )
     print("----------------Test the entire test set")
     print("Correct answers:","\n", test_Output.iloc[:])
     print("Predicted answers:", "\n", test.predict(test_Input))
     print("Accuracy: ",accuracyOfActualVsPredicted(test_Output.iloc[:], test.predict(test_Input)))  
     
def cross_val_score_manual(model, inputDF, outputSeries, k, verbose):
    numberOfElements = inputDF.shape[0]
    foldSize = numberOfElements / k 
    result = []
    for i in range(k):
        start = int(i*foldSize)
        upToNotIncluding = int((i+1)*foldSize)
        trainInputDF = pd.concat([inputDF.iloc[:start,:], inputDF.iloc[upToNotIncluding:,:]])
        trainOutputSeries = pd.concat([outputSeries.iloc[:start],outputSeries.iloc[upToNotIncluding:]])
        testInputDF = inputDF.iloc[start:upToNotIncluding,:]
        testOutputSeries = outputSeries.iloc[start:upToNotIncluding]
        if (verbose):
            print("================================") 
            print("Iteration:", i)
            print("Train input:\n", list(trainInputDF.index)) 
            print("Train output:\n", list(trainOutputSeries.index)) 
            print("Test input:\n", testInputDF.index)
            print("Test output:\n", testOutputSeries.index)
            print("================================") 
        model.fit(trainInputDF, trainOutputSeries)
        output = model.predict(testInputDF)
        result.append(accuracyOfActualVsPredicted(testOutputSeries, output))   
    return result

def testCVManual(model, k):
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:,outputCol]
    accuracies = cross_val_score_manual(model, inputDF, outputSeries, k, True)
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))

def testCVBuiltIn(model, k): 
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:,outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries, cv = k, scoring = scorer)
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))

def compareFolds():
    instance2 = OneNNClassifier()
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:,outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    print("Mean accuracy for k=3", np.mean(model_selection.cross_val_score(instance2, inputDF, outputSeries, cv = 3, scoring = scorer )))    
    print("Mean accuracy for k=10", np.mean(model_selection.cross_val_score(instance2, inputDF, outputSeries, cv = 10, scoring = scorer )))    

def standardize(df, columns):
    df.loc[:,columns] = (df.loc[:,columns] - df.loc[:,columns].mean())/df.loc[:,columns].std()
    return df

def testStandardize():
    df, inputCols, outputCol = readData()
    print("Before standardization, first 5 rows:")
    print(df.loc[:5,['Ash','Alcalinity of Ash','Magnesium']])
    df = standardize(df, ['Ash','Alcalinity of Ash','Magnesium'])    
    print("After standardization, first 5 rows:")
    print(df.loc[:5,['Ash','Alcalinity of Ash','Magnesium']])
    print("Mean are approx 0:")    
    print(df.loc[:,['Ash','Alcalinity of Ash','Magnesium']].mean())
    print("Stds are approx 1:")
    print(df.loc[:,['Ash','Alcalinity of Ash','Magnesium']].std())
    
def normalize(df, columns):
    df.loc[:,columns] = (df.loc[:,columns] - df.loc[:,columns].min())/(df.loc[:,columns].max()-df.loc[:,columns].min())
    return df

def testNormalize():
    df, inputCols, outputCol = readData()
    print("Before normalization, first 5 rows:")
    print(df.loc[:5,['Ash','Alcalinity of Ash','Magnesium']])
    normalize(df,['Ash','Alcalinity of Ash','Magnesium'])    
    print("After normalization, first 5 rows:")
    print(df.loc[:5,['Ash','Alcalinity of Ash','Magnesium']])
    print("Maxs are 1:")    
    print(df.loc[:,['Ash','Alcalinity of Ash','Magnesium']].max())
    print("Mins are 0:")
    print(df.loc[:,['Ash','Alcalinity of Ash','Magnesium']].min())
    
def comparePreprocessing():
    instance2 = OneNNClassifier()
    df, inputCols, outputCol = readData()
    print(df)
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:,outputCol]
    inputDFcopy = inputDF.copy()
    inputDFcopy1 = inputDF.copy()
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    print("The original, unaltered dataset:", np.mean(model_selection.cross_val_score(instance2, inputDF, outputSeries, cv = 10, scoring = scorer )))      
    inputDfNormalized = normalize(inputDFcopy, inputCols)
    print("The normalized dataset:", np.mean(model_selection.cross_val_score(instance2, inputDfNormalized, outputSeries, cv = 10, scoring = scorer )))    
    inputDfStandardized = standardize(inputDFcopy1, inputCols)
    print("The standardize dataset:", np.mean(model_selection.cross_val_score(instance2, inputDfStandardized, outputSeries, cv = 10, scoring = scorer )))
    
    # The original, unaltered dataset: 0.7477124183006536
    #The normalized dataset: 0.949673202614379
    #The standardize dataset: 0.9552287581699346
    #As columns in the unaltered data  have different scales, the data with higher standard deviation will influence the results more compared to the dataset with low standard deviation.
    #Thus, the accuracy of unaltered data is the lowest.
    #By standardizing the datasets, we "standardize" the scales for different columns, thus increasing the accuracy.
    #96.1%. z-transformed data means that the data is in Normal distribution.
    #leave-one-out leaves one fold for testing the dataset and uses the remaining folds to train the dataset. 
    #The results for 1-NNN is higher than my results as 1-NN uses more data for it's training set, which improves the accuracy.
    
#----------------------------------------------    

 
def visualization():
    fullDF,inputCols, outputCol= readData()
    standardize(fullDF, inputCols)
    sns.displot(fullDF.loc[:,"Malic Acid"])
    print(fullDF.loc[:, "Malic Acid"].skew())
    sns.displot(fullDF.loc[:,"Alcohol"])
    print(fullDF.loc[:, "Alcohol"].skew())
    sns.jointplot(x="Ash", y="Magnesium", data=fullDF.loc[:, ["Ash", "Magnesium"]], kind="kde")
    sns.pairplot(fullDF, hue=outputCol)
    plt.show()
    
def testSubset():
    fullDF, inputCols, outputCol= readData()
    instance3= OneNNClassifier()
    inputDF1= fullDF.loc[:,["Diluted","Proline"]] 
    inputDF2= fullDF.loc[:, ["Nonflavanoid Phenols", "Ash"]]
    outputSeries= fullDF.loc[:,outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    inputDFStandardized = standardize(inputDF1, ["Diluted", "Proline"])
    print("The accuracy using Diluted and Proline: ", np.mean(model_selection.cross_val_score(instance3,inputDFStandardized, outputSeries, cv=10, scoring=scorer)))          
    inputDFStandardized2 = standardize(inputDF2, ["Nonflavanoid Phenols", "Ash"])      
    print("The accuracy using Nonflavanoid Phenols and Ash: ", np.mean(model_selection.cross_val_score(instance3,inputDFStandardized2, outputSeries, cv=10, scoring=scorer)))
    #The accuracy using Diluted and Proline:  0.8705882352941178
    #The accuracy using Nonflavanoid Phenols and Ash:  0.5212418300653596
 
class kNNClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, k = 1):
        inputDF = None
        outputSeries = None
        self.k = k
    
    def fit(self, inputDF, outputSeries):
        self.inputDF= inputDF
        self.outputSeries=outputSeries
        
    def __predictOfKNearest(self, testInput):
        inputDF3 = self.inputDF
        testInput2 = testInput
        outputSeries2 = self.outputSeries
        xOutput=pd.Series(data=None, dtype='float64')
        for i in range(self.k):
            location= findNearestHOF(inputDF3, testInput2)
            nearest= outputSeries2.loc[location]
            xOutput = pd.concat([xOutput, pd.Series(nearest)])
            inputDF3= inputDF3.drop(location, axis=0)
            outputSeries2= outputSeries2.drop(location, axis=0)
        return xOutput.mode()[0]
        
    def predict(self, testInput):
         if isinstance(testInput, pd.core.series.Series):
             return (self.__predictOfKNearest(testInput))
         else:
             series=testInput.apply(lambda row: self.__predictOfKNearest(row),axis=1)
             return series
    
def testKNN():
    df, inputCols, outputCol= readData()
    inputDF= df.loc[:, inputCols]
    outputSeries= df.loc[:, outputCol]
    model1= kNNClassifier(1)
    model2=kNNClassifier(8)
    scorer=make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    print("Unaltered dataset, 1NN, accuracy",np.mean(model_selection.cross_val_score(model1,inputDF, outputSeries, cv=10, scoring=scorer)))
    inputDFStandardized1= standardize(inputDF, inputCols)      
    print("Standardized dataset, 1NN, accuracy ", np.mean(model_selection.cross_val_score(model1,inputDFStandardized1, outputSeries, cv=10, scoring=scorer)))
    print("Standardized dataset, 8NN, accuracy ", np.mean(model_selection.cross_val_score(model2,inputDFStandardized1, outputSeries, cv=10, scoring=scorer)))

def paramSearchPlot():
    df, inputCols, outputCol= readData()
    inputDF= df.loc[:,inputCols] 
    outputSeries= df.loc[:,outputCol]
    inputDFStandardized1= standardize(inputDF, inputCols)
    neighborList=pd.Series([1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,40,50,60,80])
    scorer=make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies=neighborList.map(lambda row: 
                                np.mean(model_selection.cross_val_score(kNNClassifier(row),inputDFStandardized1, outputSeries, cv=10, scoring=scorer)))
    print(accuracies)
    plt.plot(neighborList, accuracies)
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
    print(neighborList.loc[accuracies.idxmax()])
    
    #0     0.955229
    #1     0.938235
    #2     0.954902
    #3     0.954575
    #4     0.960131
    #5     0.960131
    #6     0.960131
    #7     0.960131
    #8     0.971569
    #9     0.977451
    #10    0.971895
    #11    0.966013
    #12    0.971569
    #13    0.971569
    #14    0.966013
    #15    0.971569
    #16    0.983007
    #17    0.983007
    #18    0.977451
    #19    0.960784
    #20    0.960784
    #21    0.932353
    #dtype: float64
    
def paramSearchPlotBuiltIn():
    df, inputCols, outputCol= readData()
    inputDF= df.loc[:,inputCols] 
    outputSeries= df.loc[:,outputCol]
    stdInputDF= standardize(inputDF, inputCols)
    alg= KNeighborsClassifier(n_neighbors=8)
    cvScores =model_selection.cross_val_score(alg ,stdInputDF, outputSeries, cv=10, scoring="accuracy") 
    print("Standardized dataset 8NN, accuracy :", np.mean(cvScores)) 
    neighborList=pd.Series([1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,40,50,60,80])
    accuracies= neighborList.map(lambda row: 
                                 np.mean(model_selection.cross_val_score(KNeighborsClassifier(row), stdInputDF, outputSeries, cv=10, scoring="accuracy"))   )
    plt.plot(neighborList, accuracies)
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
  