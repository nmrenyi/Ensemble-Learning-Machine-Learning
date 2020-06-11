# 2018011423 Ren Yi

from MyBagging import MyBaggingRegressor
from MyAdaBoost import MyAdaBoostRegressor
from MyBaggingParallel import MyBaggingParallelRegressor
from MLP import MyMLP
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import numpy as np

if __name__ == '__main__':  

    # load data and configure the choices
    train_data_df = pd.read_csv('./thuml2020/train.csv', sep = '\t')
    base_estimator_pool = [DecisionTreeRegressor(max_depth=11), LinearSVR(epsilon=0.30), MyMLP(epoch=8)]
    mode = int(input('Please input exec mode, input 0 for local train and test, 1 for output predicted file: '))
    base_estimator_choice = int(input('Please choose base estimator, input 0 for Decision Tree, 1 for SVM, 2 for MLP: '))
    ensemble_choice = int(input('Please choose ensemble method, input 0 for Serial Bagging, 1 for Adaboost, 2 for Parallel Bagging, 3 for Single Estimator: '))
    if ensemble_choice != 3:
        ensemble_num = int(input('Please input ensemble number: '))
    else:
        ensemble_num = 1
    ensemble_method_dict = {0:'Serial Bagging', 1:'Adaboost', 2:'Parallel Bagging', 3:'Single'}
    mode_dict = {0:'Local Test Mode', 1:'Output Predicted Mode'}
    base_learner_dict = {0:'DTree', 1:'SVM', 2:'MLP'}
    if ensemble_choice == 0:
        ensemble_method = MyBaggingRegressor(base_estimator_pool[base_estimator_choice], n_estimators=ensemble_num)
    elif ensemble_choice == 1:
        ensemble_method = MyAdaBoostRegressor(base_estimator_pool[base_estimator_choice], n_estimators=ensemble_num)
    elif ensemble_choice == 2:
        ensemble_method = MyBaggingParallelRegressor(base_estimator_pool[base_estimator_choice], n_estimators=ensemble_num)
    else:
        ensemble_method = base_estimator_pool[base_estimator_choice]
    if mode == 0:
        mid = int(0.8 * len(train_data_df))
        train_df = train_data_df[0:mid]
        test_df = train_data_df[mid:len(train_data_df)]
    else:
        train_df = train_data_df
        test_df = pd.read_csv('./thuml2020/test.csv', sep='\t')

    # Vectorize text with bag of words
    print('Vectorizing text with BOW model...')
    train_label = train_df['overall'].to_list()
    train_text = [str(x) + '' + str(y) for x, y in zip(train_df['summary'], train_df['reviewText'])]
    
    # select top 10000 frequently appeared words as feature and exclude stop words
    train_vectorizer = CountVectorizer(max_features=10000, stop_words='english')
    train_matrix = train_vectorizer.fit_transform(train_text)
    feature_words = train_vectorizer.get_feature_names()
    print('train_matrix size:[%d, %d]' % (len(train_text), len(feature_words)))

    test_text = [str(x) + '' + str(y) for x, y in zip(test_df['summary'], test_df['reviewText'])]
    test_vectorizer = CountVectorizer(vocabulary=feature_words)
    test_matrix = test_vectorizer.fit_transform(test_text)
    print('test_matrix size:[%d, %d]' % (len(test_text), len(feature_words)))

    # print ensemble info
    print('='*30, 'Ensemble Info', '='*30)
    print('Ensemble Method:', ensemble_method_dict[ensemble_choice])
    print('Mode:', mode_dict[mode])
    print('Ensemble Number:', ensemble_num)
    print('Base Estimator Info:', base_estimator_pool[base_estimator_choice])
    print('='*74)

    # start to train
    print('Start ensemble training...')
    start = time()
    ensemble_method = ensemble_method.fit(train_matrix, train_label)
    print('Training finished, time cost = %.2fs' % (time() - start))

    # give a prediction
    predicted = ensemble_method.predict(test_matrix)
    if mode == 0:
        test_label = test_df['overall'].to_list()
        print('RMSE:', sqrt(mean_squared_error(test_label, predicted)))
    else:
        id = list(np.arange(10001))
        id.pop(0)
        path = f'{ensemble_method_dict[ensemble_choice]}-{base_learner_dict[base_estimator_choice]}-{ensemble_num}.csv'
        pd.DataFrame({'id':id, 'predicted':predicted}).to_csv(path, index=False)
        print('Test Result File Saved To ', path)
