# Model.py
# -- Packages 
from joblib import dump, load
import os
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1500)
pd.set_option('display.max_colwidth', None)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
# - scikit-learn 
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from nltk import NaiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# - home made packages
from utils import loadModel,saveModel, loadData, label_encoding,  CreatePipeline, Makedir
from Data_preprocessing import Tokenizer_simple, Tokenizer_stemm, spacyTokenizer_lemma
from Metrics_utils import ClassificationMetrics, ROCcurve, SaveConfusionMatrix, KfoldCrossValidation, spacy_word2vec_Matrix, WordVectorTransformer


plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g','pink' ,'r','m', 'gold', 'c', 'k','orange','lime','coral','dodgerblue','gray','deeppink','peru','violet','teal','darkseagreen','firebrick'])*cycler('linestyle', ['-'])))

if __name__=="__main__":
    # --------------#
    #     Settings  #
    # --------------#
    # -- Directories/files
    DIR = os.getcwd()
    data_filename = 'comments_train.csv'
    comment_col = 'comment' # name of the column containing raw reviews
    initial_model_path = os.path.join(DIR, 'sentiment_pipe.joblib')
    data_DIR = os.path.join(DIR, data_filename)
    output_DIR= os.path.join(DIR,"output")
    fig_DIR = os.path.join(output_DIR,"Figs")
    output_filename = 'results_gridsearchCV_f1score.csv'
    output_path = os.path.join(output_DIR,output_filename )
    # -- Initiatization parameters | df | dictionnaries
    n_splits = 5 # for Stratified gridsearchCV
    random_state = 0
    test_size = 0.3 
    # To store Results 
    Results_grid ={}
    df_grid = pd.DataFrame(columns=['Model','Pipeline','Best_score','Best_params'])

    # -- Initialize Figure for ROC curves
    plt.plot(figsize=(32,32))

    # -----------------#
    #       DATA       #
    # -----------------#

    # -- Load Data from disk 
    df  = loadData(data_path)
    # -- Label distribution 
    print(f"[DATA] distribution: \n {df.groupby('sentiment').size()} \n")
    # -- Label encoding ( 1: positif, 0: negatif)  return a dataframe 
    data = label_encoding(df, 'sentiment')

    # -- Split data
    #X = data.drop(['label'],axis=1).iloc[:,0]  # if X is not a dataSerie with (x,) shape it doesn't work (x,1)  
    X = data[comment_col]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"[MODEL] SPLIT --> X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Numbers of train instances by class: {np.bincount(y_train)}")
    print(f"Numbers of test instances by class: {np.bincount(y_test)}")

    # ---------------------#
    #      Tokenizers      #
    # ---------------------#
    # -- Tokenizers: list if tokenizers to try
    tokenizers = { 
            #'No tokenizer' : None,\
            #'Simple tokenizer': Tokenizer_simple, 
            "Stemm tokenizer": Tokenizer_stemm,
            "Spacy Lemma tokenizer" :spacyTokenizer_lemma
            }

    for tokenizer_name, tokenizer in tokenizers.items():
         # --------------------#
        #      Vectorizers     #
        # ---------------------#
        # -- Vectorizers: list of vectorizers to try
        countvectorizer = CountVectorizer(tokenizer = tokenizer)
        bigram_vectorizer = CountVectorizer(tokenizer = tokenizer, ngram_range=(1, 2))
        tfidf = TfidfVectorizer(tokenizer = tokenizer)
        bigram_tfidf = TfidfVectorizer(tokenizer = tokenizer,ngram_range=(1, 2))
        Vectorizers = {
               # "Count Vectorizer": countvectorizer, \
                #"TF-IDF": tfidf,
                #"Count Vectorizer bigram": bigram_vectorizer, 
                "TF-IDF bigram":bigram_tfidf,
                }
        for vectorizer_name, vectorizer in Vectorizers.items():    
            # -------------------#
            #     Classifiers    #
            # -------------------#
            # -- Classifiers : list of classifiers to try
            SVM = SVC(random_state=random_state, probability=True)
            linearSVC = SVC(random_state=random_state, probability=True, kernel='linear')
            SGD =  SGDClassifier(max_iter=1000)
            RF = RandomForestClassifier(random_state=random_state)
            LogR = LogisticRegression(random_state=random_state, max_iter=1000)
            NaiveBayes= MultinomialNB()
            NaiveBayesBernouilli = BernoulliNB()
            classifiers = {
                #'SVM': SVM, 
                'Linear SVC' : linearSVC,
                'SGDClassifier': SGD, 
                #'Random Forest':RF, 
                #'Logistic Regression': LogR, 
                #'Naive Bayes Multinomial':  NaiveBayes
                #'Naive Bayes Bernouilli': NaiveBayesBernouilli, 
                }

            # -- Define tuning parameters for gridsearchCV
            tuned_params={
               'Logistic Regression' : {'clf__penalty':['l1', 'l2'], 'clf__C':np.logspace(-2, 3, 6), 'clf__class_weight':['balanced',None]},
               'Random Forest':{'clf__n_estimators' : [10,50,100],  'cfl__max_depth': [10, 30, 50, 70, 100, None],
                    "clf__min_samples_split" : [2,4,8], "clf__bootstrap": [True, False], 'clf__min_samples_leaf': [1, 2, 4] },
                'Linear SVC':{'clf__C': np.logspace(-2, 3, 6), 'clf__class_weight':['balanced',None]},
                'SVM': {'clf__C': np.logspace(-2, 3, 6), 'clf__gamma': np.logspace(-4, 1, 6),'clf__class_weight':['balanced',None]},
                'SGDClassifier' :{"clf__loss" : ["hinge", "log"],"clf__alpha" : [0.0001, 0.001 , 0.005, 0.01 , 0.1 ] ,"clf__penalty" : ["l2", "l1"]},
                } 
        

            # -- Cross validation
            SKF= StratifiedKFold(n_splits= 10)

            # -- Loop over classifiers
            for classifier_name, classifier in classifiers.items():
                # -- Give a name to the current model
                model_name = tokenizer_name +" + "+vectorizer_name+" + "+classifier_name
                print(f'[Current Classifier] --> {model_name}')
                # -- Pipeline Creation
                pipe = CreatePipeline(vectorizer,classifier)
                # --------------------------------------------#
                #  GridserachCV => find best hyperparameters  #
                # --------------------------------------------#
                if classifier_name in tuned_params.keys():
                    Results_grid_metrics= runGridSearchCV(pipe, X_train,y_train,tuned_params[classifier_name],SKF)
                    Results_grid.update(Results_grid_metrics)
                    Results_grid['Model'] = model_name 
                    Results_grid['Pipeline']= str(pipe)

                    # -- add results to dataframe 
                    df_grid = df_grid.append(Results_grid, ignore_index=True)
                
                    # -- to plot scorers evolution with parameters 
                    if classifier_name == 'SGDClassifier':
                        plotGridSearchCVscorers(Results_grid_metrics['cv_results'], "param_clf__alpha",os.path.join(fig_path,model_name+'_'+'alpha'+ '_GridSearchCV.png'))
                    else:    
                        try:
                            plotGridSearchCVscorers(Results_grid_metrics['cv_results'], "param_clf__gamma",os.path.join(fig_path,model_name+'_'+'gamma'+'_GridSearchCV.png'))
                        except:
                            pass
                        try:
                            plotGridSearchCVscorers(Results_grid_metrics['cv_results'], "param_clf__C",os.path.join(fig_path,model_name+'_'+'C'+'_GridSearchCV.png'))
                        except:
                            pass
                    
                    print('--------------------------------------')
                   
          
    # --------------------------#
    #     SAVE RESULTS AS CSV   #
    # --------------------------#     
    # -- save df_results
    df_grid.to_csv(output_file, index=False, header = True, sep = ';', encoding = 'utf-8') 
    df_grid.set_index('Model', inplace =True)
    print(df_grid.drop(['Pipeline'], axis=1).sort_values(by=['Best_score'], ascending=False))





