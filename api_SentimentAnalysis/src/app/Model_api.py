# 1. Library imports
import pandas as pd 
from joblib import load
from pydantic import BaseModel,BaseSettings 
import os
 
Initial_model= 'sentiment_pipe.joblib'
BERT_model = 'Camembert_training_sentiment_analysis_best_model_saved_weights.pt'
ML_model =  'SpacyLemmatokenizer-TF-IDFbigram-linearSVM_Best_model.joblib'


# -- Model Settings 
class SettingsModel(BaseSettings):
    MODEL_DIR: str = os.path.join(os.getcwd(), '..','..', 'models')
    MODEL_name: str = ML_model

model_settings = SettingsModel()

class Model:
   # -- loads the model
    def __init__(self, path):
        self.path = path 
        self.model = load(self.path)
        print(self.model)
    # -- Make a prediction based on the user-entered data
    # - Returns the predicted sentiment with its respective probability
    def predict(self, text):
        prediction = self.model.predict(text)[0]
        sentiment = "Positif" if prediction == 1 else "Négatif"
        probability = self.model.predict_proba(text)[0][1] if prediction == 1 else  self.model.predict_proba(text)[0][0] 
        print(sentiment)
        print(probability)
        return sentiment, float("{:.2f}".format(probability * 100))

model = Model(os.path.join(model_settings.MODEL_DIR, model_settings.MODEL_name))

def get_model():
    return model