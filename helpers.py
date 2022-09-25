from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

""" Inputs:
        df: dataframe
        x: list of feature names, continuous variables
    Output:
        Plots each x, to see distribution
"""
def plot_dist(df, features):

    plt.figure()
    n = len(features)
    for f, i in zip(features, range(1, n+1)):
        x = df[f]
        plt.subplot(ceil(n/2), 2, i)
        plt.plot(x)
        plt.xlabel(f)
    plt.subplots_adjust(hspace=1.2, wspace=0.5)


"""Inputs:
        df: dataframe
        features: list of column names, categorical variables
    Outputs:
        plots the count of distinct values of each column
"""
def plot_count(df, features):
    plt.figure()
    n = len(features)
    for f, i in zip(features, range(1, n+1)):
        plt.subplot(ceil(n/2), 2, i)
        df[f].value_counts().plot(kind='bar', title=f, rot=0)

    plt.subplots_adjust(hspace=1.2, wspace=0.5)


"""Inputs:
    df: the dataframe to split 
    target_name: target variable (y) column name
    test_size: size of test set as a percentage of total
    random_state: seed
    stratify: bool

    Outputs: X_train, Y_train, X_test, Y_test
"""
def split(df, target_name, test_size=0.2, random_state=0, stratify=None):
    train_df, test_df = [None, None]

    if stratify is None:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, 
        random_state=random_state, stratify=df[target_name])

    X_train = train_df.drop(columns=[target_name])
    Y_train = train_df[target_name]
    X_test = test_df.drop(columns=[target_name])
    Y_test = test_df[target_name]

    print(f'Number of columns in X_train: {len(X_train.columns)}, X_test: {len(X_test.columns)}')
    print(f'Target variable name in Y_train: {Y_train.name}, Y_test: {Y_test.name}')
    print(f'Number of training samples: {len(Y_train)}, testing samples: {len(Y_test)}')

    return [X_train, Y_train, X_test, Y_test]


"""Inputs:
        model: the trained model
        xtrain, xtest, ytrain, ytest: the corresponding x and y dataframes used for training and test.

    Outputs: prints the training and testing metrics like accuracies, confusion matrix, precision etc., 
"""
def print_metrics(model, xtrain, xtest, ytrain, ytest):
    # make predictions for test data
    y_pred = model.predict(xtest)
    y_pred_train = model.predict(xtrain)

    if not isinstance(y_pred[0], np.bool_ ):
        y_pred = [round(value) for value in y_pred]
        y_pred_train = [round(value) for value in y_pred_train]

    # evaluate predictions
    accuracy_test = accuracy_score(ytest, y_pred)
    print("Test Accuracy: %.2f%%" % (accuracy_test * 100.0))
    accuracy_train = accuracy_score(ytrain, y_pred_train)
    print("Train Accuracy: %.2f%%" % (accuracy_train * 100.0))
    print("")

    # confusion matrix 
    cm = confusion_matrix(ytest, y_pred)
    cmtx = pd.DataFrame(cm, index=['actual:no', 'actual:yes'], columns=['pred:no', 'pred:yes'])
    print("Confusion matrix:")
    print(cmtx)
    print("")

    tn, fp, fn, tp = cm.ravel()
    print(f'True Negatives: {tn}, False Positives: {fp}, False negatives: {fn}, True positives: {tp}')
    print("")

    eval_df = pd.DataFrame(precision_recall_fscore_support(ytest, y_pred), 
    index=["Precision", "Recall", "Fscore", "Support"], columns=["Class 0", "Class 1"])
    print(eval_df)
    print("")



"""Inputs:
        Takes fitted model with eval set [(x_train, y_train), (x_test, y_test)]
    Outputs: plots the evaluation metrics per epoch
"""
def plot_metrics(model):
    # performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # plot log loss for train and test
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.title('XGBoost Log Loss')
    plt.show()

    # plot AUC for train and test
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.title('XGBoost AUCROC')
    plt.show()

    print('Test ROC AUC: %.3f' % results['validation_1']['auc'][-1])
    print('Test logloss: %.3f' % results['validation_1']['logloss'][-1])