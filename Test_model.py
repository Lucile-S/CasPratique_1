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
from sklearn.model_selection import train_test_split
# - home made packages
from utils import loadModel,saveModel, loadData, label_encoding,  CreatePipeline, Makedir
from Data_preprocessing import Tokenizer_simple, Tokenizer_stemm, spacyTokenizer_lemma
from Metrics_utils import  Test_ClassificationMetrics, ROCcurve, SaveConfusionMatrix

plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g','pink' ,'r','m', 'gold', 'c', 'k','orange','lime','coral','dodgerblue','gray','deeppink','peru','violet','teal','darkseagreen','firebrick'])*cycler('linestyle', ['-'])))

if __name__=="__main__":
    # --------------#
    #     Settings  #
    # --------------#
    # -- Directories/files
    DIR = os.getcwd()
    Data_DIR = os.path.join(DIR, "Data")
    Model_DIR =os.path.join(DIR,'Models')
    output_DIR = os.path.join(DIR,"output")
    Makedir(output_DIR)
    fig_DIR = os.path.join(output_DIR,"Figs")
    Makedir(fig_DIR)
    # Name of our csv file containing reviews
    data_filename = 'comments_train.csv'
    data_path = os.path.join(Data_DIR, data_filename)
    # Name of the column containing raw reviews
    comment_col = 'comment' 
    # path of the model 
    model_path = os.path.join(Model_DIR, 'SpacyLemmatokenizer-TF-IDFbigram-linearSVM_Best_model.joblib')
    model_name ='SpacyLemmatokenizer-TF-IDFbigram-linearSVM'

    # -- Initialize Figure for ROC curves
    plt.plot(figsize=(32,32))
    ROC_plot_filename = 'ROC_curves_test.png'

    # -- Load Model initial from disk 
    pipeline = loadModel(model_path)
    # - Uncomment to get pipeline parameters 
    #print(pipeline.get_params())

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
    X_test = data[comment_col]
    y_test = data['label']
    print(f"[X_TEST] X_test: {X_test.shape}")
    print(f"Numbers of test instances by class: {np.bincount(y_test)}")

    # --------------------------#
    #  Evaluation on TEST set   #
    # --------------------------#   
    print(' \n ------------- TEST Evaluation ------------- ')             
    # -- calcul metrics 
    Results_metrics = Test_ClassificationMetrics(pipeline,X_test,y_test)
    # -- Save confusion matrix
    SaveConfusionMatrix(pipeline, model_name, X_test, y_test, os.path.join(fig_DIR,model_name+ '.png'))
    # -------------------#
    #     ROC curves     #
    # -------------------#
    # -- ROC curve plots
    ROCcurve(pipeline,model_name, X_test,y_test)
    # -- Save the ROC figure 
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("ROC curves")
    plt.savefig(os.path.join(fig_DIR,ROC_plot_filename), bbox_inches='tight')  

            


