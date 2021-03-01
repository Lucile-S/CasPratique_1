
import os
import pandas as pd
import numpy as np
from joblib import dump, load 

# - scikit-learn 
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline, Pipeline



def saveModel(model, path):
    dump(model, path)

def loadModel(path):
    print(f"[MODEL] {load(path)}")
    return load(path)

def loadData(path):
    df = pd.read_csv(path)
    print(f"[DATA] shape: {df.shape} \n")
    print(f"[DATA] head: \n {df.head()} \n")
    return df 

def Makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def label_encoding(df, col='label'):
    """ input : dataframe ; output : dataframe with binarized labeled column renamed 'label' """
    encoder = LabelBinarizer().fit(df[col])
    label_df = pd.DataFrame(encoder.transform(df[col]), columns=['label'])
    data = pd.concat([df, label_df], axis=1).drop([col], axis=1)
    return data

def CreatePipeline(vectorizer, classifier):
    """ Create a pipeline with a vectorizer and a classifier """
    pipe = Pipeline([('vectorizer', vectorizer), ('clf', classifier)])
    return pipe



