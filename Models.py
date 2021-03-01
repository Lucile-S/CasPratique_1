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
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    Data_DIR= os.path.join(DIR,'Data')
    data_filename = 'comments_train.csv'
    comment_col = 'comment' # name of the column containing raw reviews
    initial_model_path = os.path.join(DIR, 'sentiment_pipe.joblib')
    data_path = os.path.join(Data_DIR, data_filename)
    output_DIR = os.path.join(DIR,"output")
    model_DIR= os.path.join(DIR, "Models")
    fig_DIR = os.path.join(output_DIR,"Figs")
    output_filename = 'results.csv'
    saved_model_filename = 'Best_model.joblib'
    ROC_plot_filename = 'ROC_curves.png'
    # -- Initiatization parameters | df | dictionnaries
    n_splits = 10 # for Stratified gridsearchCV
    random_state = 0
    test_size = 0.3 
    selection_metric = 'eval_F1score' # metric by which the best model will be selected 
    # -- To store results 
    Results = {}
    # for gridsearchCV
    Results_grid ={}

    # -- Dataframe where results will be saved
    df_results = pd.DataFrame(columns = ['Model','Pipeline','CV_train_score', 'eval_score','eval_precision', 'eval_recall', 'eval_F1score','eval_AUC','train_score','test_score','precision','recall', 'F1score','AUC','AUC_proba'])

    # -- Load Model initial from disk 
    pipeline_0 = loadModel(initial_model_path)
    # - Uncomment to get pipeline parameters 
    #print(pipeline_0.get_params())

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

    # -------------------------#
    #       Initial Model      #
    # -------------------------#
    # -- Initial Model Evaluation 
    print(' \n ------------- Initial Model Evaluation ------------- ')
    Results['Pipeline'] = str(pipeline_0)
    Results['Model']= "Initial Model"   
    Results_metrics = ClassificationMetrics(pipeline_0,X_train,y_train,X_test,y_test)
    Results.update(Results_metrics)
    # - Initial Model confusion Matrix 
    SaveConfusionMatrix(pipeline_0, "Initial Model" , X_test, y_test, os.path.join(fig_DIR,"Initial Model" + '.png'))
    # - Initial model ROC curve 
    ROCcurve(pipeline_0,"Initial Model", X_test,y_test)
    #df_results = df_results.append(Results, ignore_index=True)  
    print('------------------------------------- \n')

    # ---------------------#
    #      Tokenizers      #
    # ---------------------#
    # -- Tokenizers: list if tokenizers to try ; comment unwanted ones 
    tokenizers = { 
            'No tokenizer' : None,\
            'Simple tokenizer': Tokenizer_simple, 
            "Stemm tokenizer": Tokenizer_stemm,
            "Spacy Lemma tokenizer" :spacyTokenizer_lemma
            }

    for tokenizer_name, tokenizer in tokenizers.items():
         # --------------------#
        #      Vectorizers     #
        # ---------------------#
        # -- Vectorizers: list of vectorizers to try ; comment unwanted ones 
        countvectorizer = CountVectorizer(tokenizer = tokenizer)
        bigram_vectorizer = CountVectorizer(tokenizer = tokenizer, ngram_range=(1, 2))
        tfidf = TfidfVectorizer(tokenizer = tokenizer)
        bigram_tfidf = TfidfVectorizer(tokenizer = tokenizer,ngram_range=(1, 2))
        Vectorizers = {
                "Count Vectorizer": countvectorizer, \
                "TF-IDF": tfidf,
                 "Count Vectorizer bigram": bigram_vectorizer, 
                "TF-IDF bigram": bigram_tfidf,
                #"Spacy W2V": WordVectorTransformer(),
                }
        for vectorizer_name, vectorizer in Vectorizers.items():    
            # -------------------#
            #     Classifiers    #
            # -------------------#
            # -- Classifiers : list of classifiers to try ; comment unwanted ones 
            SVM = SVC(random_state=random_state, probability=True)
            linearSVC = SVC(random_state=random_state, probability=True, kernel='linear')
            SGD =  SGDClassifier( random_state=random_state,max_iter=1000)
            RF = RandomForestClassifier(random_state=random_state)
            LogR = LogisticRegression(random_state=random_state, max_iter=1000)
            NaiveBayes= MultinomialNB()
            NaiveBayesBernouilli = BernoulliNB()
            # Best model obtained after testing 
            BestModel = SVC(C= 1, class_weight=None,random_state=random_state, probability=True, kernel='linear') 
            classifiers = {
                'SVM': SVM, \
                'Linear SVC' : linearSVC,
                'SGDClassifier': SGD, 
                'Random Forest':RF, 
                'Logistic Regression': LogR, 
                'Naive Bayes Multinomial':  NaiveBayes,
                'Naive Bayes Bernouilli': NaiveBayesBernouilli, 
                'Best Model': BestModel,
                }

            # -- Cross validation method
            SKF= StratifiedKFold(n_splits= n_splits)

            # -- Initialize best score
            Best_score = 0.0 

            # -- Loop over classifiers
            for classifier_name, classifier in classifiers.items():
                # -- Give a name to the current model
                model_name = tokenizer_name +" + "+vectorizer_name+" + "+classifier_name
                print(f'Current Classifier --> {model_name}')
                # -- Pipeline Creation
                pipe = CreatePipeline(vectorizer,classifier)
                # --------------------------------------------#
                #  Cross Validation performance evaluation    #
                # --------------------------------------------#
                Results_CV_metrics = KfoldCrossValidation(pipe, X_train,y_train, SKF)
                # -- Store cv results into the results dictionnary 
                Results.update(Results_CV_metrics)
                # --------------------------#
                #  Evaluation on TEST set   #
                # --------------------------#                
                pipe.fit(X_train, y_train)
                Results['Model'] = model_name 
                Results['Pipeline']= str(pipe)
                # -- calcul metrics 
                Results_metrics = ClassificationMetrics(pipe,X_train,y_train,X_test,y_test)
                # -- Store them into the results dictionanry 
                Results.update(Results_metrics)
                # -- add results to dataframe 
                df_results = df_results.append(Results, ignore_index=True)
                # -- Save confusion matrix
                SaveConfusionMatrix(pipe, model_name, X_test, y_test, os.path.join(fig_DIR,model_name+ '.png'))
                # -- ROC curve plots
                ROCcurve(pipe,model_name, X_test,y_test)
                print('--------------------------------------')

                # ---------------------------#
                #    Best Model Selection    #
                # ---------------------------#
                # model selection based on selection_metric score 
                if Results[selection_metric] > Best_score:
                    Best_score = Results[selection_metric]
                    Best_model = pipe
                    Best_model_name = model_name
                
            # -------------------#
            #     ROC curves     #
            # -------------------#
            # -- Save the ROC figure 
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.title("ROC curves")
            plt.savefig(os.path.join(fig_DIR,ROC_plot_filename), bbox_inches='tight')  

         
    # --------------------------#
    #     SAVE RESULTS AS CSV   #
    # --------------------------#     
    # -- Save df_results
    Makedir(output_DIR)
    df_results.to_csv(os.path.join(output_DIR, output_filename), index=False, header = True, sep = ';', encoding = 'utf-8') 
    df_results.set_index('Model', inplace =True)
    print(df_results)
    print(df_results.drop(['Pipeline'], axis=1).sort_values(by=['F1score'], ascending=False).iloc[:,7:])
    print(df_results.drop(['Pipeline'], axis=1).sort_values(by=['eval_F1score'], ascending=False).iloc[:,:7])
    
    # --------------------------#
    #     SAVE Best MODEL       #
    # --------------------------#  
    # -- Just run the script with the best model obtained (comment unwanted ones)
    Makedir(model_DIR)
    print(f'[BEST MODEL] {Best_model_name}  pipeline saved as best model')
    print(f'\n {Best_model}')
    saveModel(Best_model,  os.path.join(model_DIR, Best_model_name.replace(' ','').replace('+','-') +'_'+ saved_model_filename))
    


