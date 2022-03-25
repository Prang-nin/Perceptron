# import important modules
import numpy as np
import pandas as pd
import random

# Load input data
test = np.array(pd.read_csv("test.data", header=None))
train = np.array(pd.read_csv("train.data",  header=None))

# drop unwanted class
def dropClass(data, label):
    '''
    prameter : data (array) , label (str)
    output : Sliced array that drop all rows that have the label = label (str)
    '''
    return data[data[:,-1] != label]
# get the features columns from dataset
def getFeature (data):
    return data[:,:-1]

# get the label column from dataset
def getLabel (data):
    return data[:,-1]

# map the class labels into binary classification
def mapBinaryClass(cleanedData, classList):
    '''
    parameters: 
        cleanedData (array) 
        classList   (list)  : list of consider classes, [positive, negative]
    output : an array that mapped positive label with 1 and negative label with -1
    '''
    mappedData = cleanedData.copy()
    for i in range(mappedData.shape[0]):
        # map label to the class feature according to the list\
        if mappedData[i,-1] == classList[0]:
            mappedData[i,-1] = 1
        elif mappedData[i,-1] == classList[1]:
            mappedData[i,-1] = -1
    return mappedData

#Activation function : Sign Function in this case threshold = 0
def activationFunction(activationScore):
    '''
    parameter : activationScore   return : 1 when activation score higer than 0 otherwise -1
    '''
    if activationScore > 0:
        return 1
    return -1

# Perceptron training algorithm 
# lamda is for L2 regularised
def perceptronTrain(train, maxIter=20, lamda = 0):
    '''
    parameters : 
        train dataset (array)  , maxIter (int) default = 20  , lamda (float) default =0
    output : tuple of bias and weights array
    '''
    # split features and the label
    features = getFeature(train)
    actualOutput = getLabel(train)
    # set the weight array that has size equal to number of features and assign 0 for each element
    weight = np.zeros(features.shape[1])
    bias = 0
    for epoch in range(maxIter):
        D = list(zip(features, actualOutput))
    # set the seed with random number and shuffle the data before pass into training algorithm
        random.seed(21)
        random.shuffle(D)
        for x, y in D:
    # calculate activation score for each instance 
            activationScore = np.dot(weight, x) + bias
    # update the weight when missclassification 
            if y*activationScore <= 0 :
                weight = ((1-2*lamda)*weight ) + y * x
                bias = bias + y
    return bias, weight

# Perceptron test function
def perceptronTest(biasWeights, featureTest):
    '''
    parameters: biasWeights (tuple) , featureTest (array)
    output: predict label (array)
    '''
    bias = biasWeights[0]
    weights = biasWeights[1]
    y_pred = []
    for x in featureTest:
        activationScore = np.dot(weights, x) +bias
        y = activationFunction(activationScore)
        y_pred.append(y)
    return np.array(y_pred)


def mapMultiClass( inputData, classList):
    '''
    # This function take data and list of labels 
    # the output will be labeled as 0,1,2,3,.. accordingly to the order of the list
    parameters: inputData (array) , classList (List)
    output : output (array)
    '''
    outputData = inputData.copy()
    label = [i for i in range(len(classList))]
    for y in range(outputData.shape[0]):
        for i in range (len(classList)):
            if outputData[y,-1] == classList[i]:
                outputData[y,-1] = label[i]
    return outputData


def multiClassTrain(train,maxIter=20, lamda = 0):
    '''
    # This function will generate one vs rest models and calculate weights and bias for each model
    and then return a stack of bias-weights array
    parameters : 
        train dataset (array)  , maxIter (int) default = 20  , lamda (float) default =0
    output : array stack of bias and weights array
    '''
    trainData = train.copy()
    classLabel = np.unique(trainData[:,-1])
    stack = []
    for i in classLabel:
        oneRest = trainData.copy()
        for y in range(oneRest.shape[0]):
            if oneRest[y,-1] == classLabel[i]:
                oneRest[y,-1] = 1
            else:
                oneRest[y,-1] = -1
        weightOneRest = perceptronTrain(oneRest, maxIter, lamda)
        stack.append(weightOneRest)
    return  np.vstack(stack).T

# multiple of weights and bias as input(from different model)
#multiweight = [ bias, (w1,w2,...), label]
def multiClassifier(multiWeights, featureTest):
    """
    parameters: multiWeights is a stack of bias-weights vector (array), featureTest (array)
    output: predict label (array)
    """
    yPred = []
    weights = multiWeights[1]
    bias = multiWeights[0]
    for x in featureTest:
        scoreArray = []
        for w, b in zip(weights, bias):
            activationScore = np.dot(w, x) + b
            scoreArray.append(activationScore)
        #get the label based on the maximum activation score and append to the prediction list
        yPred.append(np.argmax(scoreArray))
    return np.array(yPred)


def confusionMatrix(actual, predicted):
    '''
    parameters : actual (array) , predicted (array)
    return confusion matrix (array)
    '''
    # extract the different classes
    feature = np.unique(actual) 
    # initialize the confusion matrix with zeros
    conMatrix = np.zeros((len(feature), len(feature))) 
    for i in range(len(feature)):
        for j in range(len(feature)): 
            conMatrix[i, j] = np.sum((actual == feature[i]) & (predicted == feature[j]))
    return conMatrix


def evaluationReport(model,actual, predicted):
    """
    This function will generate the evaluation report using confusion matrix
    the report consists of accurancy, precision, recall, F-score, Macro F-Score
    """
    confusion = confusionMatrix(actual, predicted)
    diagnoalSum = confusion.trace()
    accurate =  diagnoalSum/confusion.sum()
    print("Model evaluation report of {} : \n accuraccy : {:.2f}".format(model,accurate))
    labelSet = [ i for i in range(len(np.unique(actual)))]
    for label in labelSet:
        if np.sum(confusion[:,label]) == 0 :
            precise = 0
        else: precise =  np.sum(confusion[label,label])/np.sum(confusion[:,label])
        if np.sum(confusion[label,:]) == 0:
            reCall = 0
        else: reCall = np.sum(confusion[label,label])/np.sum(confusion[label,:])
        if (precise+reCall) == 0:
            fScore = 0
        else: fScore = (2*precise*reCall)/(precise+reCall)
        macroF = fScore/len(labelSet)
        print(" label :{}  precision : {:.2f} recall : {:.2f}  F-score : {:.2f}  MacroF-score : {:.2f}".format(label,precise,reCall,fScore,macroF))


if __name__ == "__main__":
    # Data-preprocess 1 Vs 2
    oneTwoTrain = mapBinaryClass(dropClass(train,'class-3'), ['class-1','class-2'])
    oneTwoTest = mapBinaryClass(dropClass(test,'class-3'), ['class-1','class-2'])
    # Split data into train and test for the algorithm 1 Vs 2
    xTrainOneTwo = getFeature(oneTwoTrain)
    yTrainOneTwo = getLabel(oneTwoTrain)
    xTestOneTwo = getFeature(oneTwoTest)
    yTestOneTwo = getLabel(oneTwoTest)
    # Train and Test the algorithm 1 Vs 2
    yPredOneTwoTrain = perceptronTest(perceptronTrain(oneTwoTrain), xTrainOneTwo)
    yPredOneTwoTest = perceptronTest(perceptronTrain(oneTwoTrain), xTestOneTwo)
    # print out the evaluation reports
    evaluationReport('One Vs Two Binary-classification for Train dataset', yTrainOneTwo, yPredOneTwoTrain)
    evaluationReport('One Vs Two Binary-classification for Test dataset', yTestOneTwo, yPredOneTwoTest)

    # Data-preprocess 1 Vs 3
    oneThreeTrain = mapBinaryClass(dropClass(train,'class-2'), ['class-1','class-3'])
    oneThreeTest = mapBinaryClass(dropClass(test,'class-2'), ['class-1','class-3'])
    # Split data into train and test for the algorithm 1 Vs 3
    xTrainOneThree = getFeature(oneThreeTrain)
    yTrainOneThree = getLabel(oneThreeTrain)
    xTestOneThree = getFeature(oneThreeTest)
    yTestOneThree = getLabel(oneThreeTest)
    # Train and Test the algorithm 1 Vs 3
    yPredOneThreeTrain = perceptronTest(perceptronTrain(oneThreeTrain), xTrainOneThree)
    yPredOneThreeTest = perceptronTest(perceptronTrain(oneThreeTrain), xTestOneThree)
    # print out the evaluation reports
    evaluationReport('One Vs Three Binary-classification for Train dataset', yTrainOneThree, yPredOneTwoTrain)
    evaluationReport('One Vs Three Binary-classification for Test dataset', yTestOneThree, yPredOneTwoTest)

    # Data-preprocess 2 Vs 3
    twoThreeTrain = mapBinaryClass(dropClass(train,'class-1'), ['class-2','class-3'])
    twoThreeTest = mapBinaryClass(dropClass(test,'class-1'), ['class-2','class-3'])
    # Split data into train and test for  the algorithm 2 Vs 3
    xTrainTwoThree = getFeature(twoThreeTrain)
    yTrainTwoThree = getLabel(twoThreeTrain)
    xTestTwoThree = getFeature(twoThreeTest)
    yTestTwoThree = getLabel(twoThreeTest)
    # Train and Test the algorithm 2 Vs 3
    yPredTwoThreeTrain = perceptronTest(perceptronTrain(twoThreeTrain), xTrainTwoThree)
    yPredTwoThreeTest = perceptronTest(perceptronTrain(twoThreeTrain), xTestTwoThree)
    # print out the evaluation reports
    evaluationReport('Two Vs Three Binary-classification for Train dataset', yTrainTwoThree, yPredTwoThreeTrain)
    evaluationReport('Two Vs Three Binary-classification for Test dataset', yTestTwoThree, yPredTwoThreeTest)

    # ignore overflow warning
    np.warnings.filterwarnings('ignore')
    # multiclass classification
    # Data-preprocessing
    multiTest = mapMultiClass(test, ['class-1','class-2','class-3'])
    multiTrain = mapMultiClass(train, ['class-1','class-2','class-3'])
    # Split data into train and test for the one-vs-rest multi-class 
    xTrainMulti = getFeature(multiTrain)
    yTrainMulti = getLabel(multiTrain)
    xTestMulti = getFeature(multiTest)
    yTestMulti = getLabel(multiTest)
  
    # calculate a stack of bias-weights vectors of each one vs rest model 
    # pass the caculated bias-weights vector into muticlass-classification function
    # Multi-class Classification when lamda = 0 by default
    yPredMultiTrain = multiClassifier(multiClassTrain(multiTrain), xTrainMulti) # lamda = 0 by default
    yPredMultiTest = multiClassifier(multiClassTrain(multiTrain), xTestMulti) # lamda = 0 by default
    # Multi-class Classification when lamda = 0.01 with 20 iteration training
    yPredMultiL1Train = multiClassifier(multiClassTrain(multiTrain,20,0.01), xTrainMulti) 
    yPredMultiL1Test = multiClassifier(multiClassTrain(multiTrain,20,0.01), xTestMulti) 
    # Multi-class Classification when lamda = 0.1 with 20 iteration training
    yPredMultiL2Train = multiClassifier(multiClassTrain(multiTrain,20,0.1), xTrainMulti) 
    yPredMultiL2Test = multiClassifier(multiClassTrain(multiTrain,20,0.1), xTestMulti) 
    # Multi-class Classification when lamda = 1 with 20 iteration training
    yPredMultiL3Train = multiClassifier(multiClassTrain(multiTrain,20,1.0), xTrainMulti) 
    yPredMultiL3Test = multiClassifier(multiClassTrain(multiTrain,20,1.0), xTestMulti) 
    # Multi-class Classification when lamda = 10 with 20 iteration training
    yPredMultiL4Train = multiClassifier(multiClassTrain(multiTrain,20,10.0), xTrainMulti) 
    yPredMultiL4Test = multiClassifier(multiClassTrain(multiTrain,20,10.0), xTestMulti) 
    # Multi-class Classification when lamda = 100 with 20 iteration training
    yPredMultiL5Train = multiClassifier(multiClassTrain(multiTrain,20,100), xTrainMulti) 
    yPredMultiL5Test = multiClassifier(multiClassTrain(multiTrain,20,100), xTestMulti)
    # print out the evaluation reports
    evaluationReport('One Vs Rest Multi-classification for train dataset', yTrainMulti, yPredMultiTrain)
    evaluationReport('One Vs Rest Multi-classification for test dataset', yTestMulti, yPredMultiTest)

    evaluationReport('One Vs Rest Multi-classification Lamda = 0.01 for train dataset ',yTrainMulti, yPredMultiL1Train)
    evaluationReport('One Vs Rest Multi-classification Lamda = 0.01 for test dataset',yTestMulti, yPredMultiL1Test)

    evaluationReport('One Vs Rest Multi-classification Lamda = 0.1 for train dataset',yTrainMulti, yPredMultiL2Train)
    evaluationReport('One Vs Rest Multi-classification Lamda = 0.1 for test dataset',yTestMulti, yPredMultiL2Test)

    evaluationReport('One Vs Rest Multi-classification Lamda = 1.0 for train dataset',yTrainMulti, yPredMultiL3Train)
    evaluationReport('One Vs Rest Multi-classification Lamda = 1.0 for test dataset',yTestMulti, yPredMultiL3Test)

    evaluationReport('One Vs Rest Multi-classification Lamda = 10.0 for train dataset',yTrainMulti, yPredMultiL4Train)
    evaluationReport('One Vs Rest Multi-classification Lamda = 10.0 for test dataset',yTestMulti, yPredMultiL4Test)

    evaluationReport('One Vs Rest Multi-classification Lamda = 100 for train dataset',yTrainMulti, yPredMultiL5Train)
    evaluationReport('One Vs Rest Multi-classification Lamda = 100 for test dataset',yTestMulti, yPredMultiL5Test)
