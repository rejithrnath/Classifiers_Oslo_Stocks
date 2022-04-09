


import datetime
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from finta import TA

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


interval_time ='1d'
lags = 3

def Classifier_ML(symbol):
    print("")
    print(symbol)
    start = "2021-01-01" 
    end = "2021-11-07"  
    data=yf.download(symbol,start,end, interval=interval_time, progress = False)
    
    print('Data First Row')
    print(data.head(1))
    print('')
    print('Data Last Row')
    print(data.tail(1))
    print('')
    print('Data Shape -> ' + str(data.shape))
    print('')
    print('Descriptive status ->')
    print(data.describe())
    
    data[data.columns.values] = data[data.columns.values].ffill()    
    
    data["returns"] = np.log(data['Close'].div(data['Close'].shift(1)))
    data["direction"]  = np.where(data['returns'] > 0, 1, 0)
    
    data.rename(columns={'Open': 'open', 'Close' :'close','High': 'high','Low':'low'}, inplace=True)

    data['adx'] = TA.ADX(data)
    data['cmo'] = TA.CMO(data)
    data['wil'] = TA.WILLIAMS(data)
    data['adl'] = TA.ADL(data)
    data['cci'] = TA.CCI(data)
    
    cols = [] 
        
    features=['adx','cmo','wil','adl','cci']

      
    for f in features:
      for lag in range(1, lags + 1):
        col = "{}_lag_{}".format(f, lag)
        data[col] = data[f].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)

    data[['close']].plot(grid=True)
    plt.savefig(symbol+'_Close.png')
    
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.suptitle(symbol, fontsize=12)
    sns.heatmap(corr_matrix)
    plt.savefig(symbol+'_corr.png')
    
    fig = plt.figure()
    plot = data.groupby(['direction']).size().plot(kind='barh', color='red')
    plt.savefig(symbol+'_direction.png')
    
    
    dataset_length = data.shape[0]
    split = int(dataset_length * 0.70)

    data = data.drop(labels=['open','close','high','low','Adj Close','Volume','returns','adx','cmo','wil','adl','cci'], axis=1)
    
    data.dropna(inplace=True)
    
 
    
    dataset_length = data.shape[0]
    split = int(dataset_length * 0.70)  
    
    X = data.copy()
    #for random forest and SVM
    X_train_rf_svm, X_validation_rf_svm = X[:split], X[split:] 
    
    #for Logistic Regression
    X_train, X_validation = X[:split], X[split:]
    mu,std = X_train.mean(),X_train.std()
    X_train = (X_train - mu) /std
    X_validation = (X_validation -mu)/std
    
    #dependent variable
    y= data["direction"]
    y_train, y_validation = y[:split], y[split:]
    
    # Random Forest
    
    ## hyper parameter tuning
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    # define models and parameters
    model = RandomForestClassifier()
    n_estimators = [500, 600, 700, 800]
    max_depth =[60, 80 , 100, 120]
    
    # define grid search
    grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    rf = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = rf.fit(X_train_rf_svm[cols], y_train)
    # summarize results
    print("Best RF : %f using %s" % (grid_result.best_score_, grid_result.best_params_))
 
    
    
    # Logistic Regression

    # define models and parameters
    model = LogisticRegression()
    solvers = ['newton-cg']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    max_iter=[1000]

    # define grid search
    grid = dict(solver=solvers,max_iter=max_iter,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    lm = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = lm.fit(X_train[cols], y_train)
    # summarize results
    print("Best LM: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    
    lm.fit(X_train[cols], y_train)
    prediction_lm =[]
    prediction_lm = lm.predict(X_validation[cols])

    
    # SVM
    # define model and parameters
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    # define grid search
    grid = dict(kernel=kernel,C=C,gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    svm = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = svm.fit(X_train[cols], y_train)
    # summarize results
    print("Best SVM: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    prediction_svm =[] 
    prediction_svm = svm.predict(X_validation[cols])
    
    


    
    #Evaluation RF
    
    prediction_rf =[]
    prediction_rf = rf.predict(X_validation_rf_svm[cols])
    
    print('Accuracy RF -> '+ str(accuracy_score(prediction_rf,
                             np.sign(y_validation))))
    print('Precision Recall Fscore Support RF -> ')
    print(precision_recall_fscore_support(y_validation, prediction_rf, average='binary'))
    
    
    
    #Evaluation LM
    print('Accuracy LM -> '+ str(accuracy_score(prediction_lm,
                             np.sign(y_validation))))
    print('Precision Recall Fscore Support LM -> ')
    print(precision_recall_fscore_support(y_validation, prediction_lm, average='binary'))
    
    
    
    #Evaluation SVM
    print('Accuracy SVM -> '+ str(accuracy_score(prediction_svm,
                             np.sign(y_validation))))
    print('Precision Recall Fscore Support SVM -> ')
    print(precision_recall_fscore_support(y_validation, prediction_svm, average='binary'))
    
    
    

with open("OL.csv") as f:  
            lines = f.read().splitlines()
           
            for symbol in lines:
              
                  # print(symbol)
                  print("")
                  print("-------------------------------------------------------------")
                  Classifier_ML(symbol)
                  print("-------------------------------------------------------------")
                  print("")