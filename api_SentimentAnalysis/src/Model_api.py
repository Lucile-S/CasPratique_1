# 1. Library imports
import pandas as pd 
from joblib import load
from pydantic import BaseModel, BaseSettings 
import os
import json
import torch
import torch.nn.functional as F
from transformers import CamembertTokenizer, CamembertForSequenceClassification



Available_models = {
    "Initial_model": 'sentiment_pipe.joblib',
    "Camembert_model" :'Camembert_training_sentiment_analysis_best_model_saved_weights.pt',
    "LemmaTokenizer_TFIDFbigram_LinearSVM": "SpacyLemmatokenizer-TF-IDFbigram-linearSVM_Best_model.joblib",
}


def GetKey(dic, val):
   for key, value in dic.items():
      if val == value:
         return key

# -- Model Settings 
class SettingsModel(BaseSettings):
    MODEL_DIR: str = os.path.join(os.getcwd(), '..', 'models')
    #MODEL_filename: str = Available_models["LemmaTokenizer_TFIDFbigram_LinearSVM"]
    MODEL_filename: str = Available_models["Camembert_model"]
    MODEL_name: str = GetKey(Available_models, MODEL_filename)
    CONFIG_file : str = os.path.join(os.getcwd(), '..','config',"BERT_config.json")
    #CONFIG_file : str = 'None'


model_settings = SettingsModel()

if model_settings.CONFIG_file != 'None':
    with open(model_settings.CONFIG_file) as json_file:
        config = json.load(json_file)


# -------- BERT MODEL --------------- #

class BertModel:
    def __init__(self,DIR, filename):
        self.path = os.path.join(DIR, filename)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CamembertTokenizer.from_pretrained(config["BERT_MODEL"])
        classifier = CamembertForSequenceClassification.from_pretrained(config['BERT_MODEL'], num_labels=len(config["CLASS_NAMES"]))
        classifier.load_state_dict(torch.load(self.path, map_location=self.device))
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def encode_text(self, texts:list):
        inputs = [] # List of token ids to be fed to a model.
        attention_masks = []
        for text in texts:
            sequence_dict = self.tokenizer.encode_plus(text, max_length=config['MAX_SEQUENCE_LEN'], pad_to_max_length=True)
            input_ids = sequence_dict['input_ids']
            att_mask = sequence_dict['attention_mask']
            inputs.append(input_ids)
            attention_masks.append(att_mask)
        return torch.tensor(inputs).to(self.device), torch.tensor(attention_masks).to(self.device)
    
    def prediction(self, text):
        input_ids, att_mask = self.encode_text([text])
        logits = self.classifier(input_ids, att_mask)
        # compute proba values (list)
        probs= F.softmax(logits[0], dim=1)[0].tolist()
        # compute pred value (0 or 1)
        pred = torch.argmax(logits[0], 1).tolist()[0]
        print(f'Sentiment prediction "{text}" --> {config["CLASS_NAMES"][pred]} | with a probability of {probs[pred]:.2f}')
        sentiment = config['CLASS_NAMES'][pred]
        probability = float("{:.2f}".format(probs[pred] * 100))
        return sentiment, probability 
        
    # -- Make a prediction based on the user-entered data : either a unique string text or a list of text 
    def get_predictions(self,text):
        # -- multiple texts prediction
        if isinstance(text, list):
            print(f"Prediction of {len(text)} texts...")
            sentiments =[]
            probabilities =[]
            for item in text:
                sentiment, probability  = self.prediction(item)
                sentiments.append(sentiment)
                probabilities.append(probability)
            return sentiments, probabilities
        # -- unique text prediction
        if isinstance(text, str):
            sentiment, probability  = self.prediction(text)
            return sentiment, probability 

# -------- ML  MODEL --------------- #
class Model:
   # -- loads the model
    def __init__(self, DIR, filename):
        self.path = os.path.join(DIR, filename)
        self.classifier = load(self.path)
        print(self.classifier)

    # - Returns the predicted sentiment with its respective probability
    def prediction(self, text):
        text = [text]
        prediction = self.classifier.predict(text)[0]
        sentiment = "Positif" if prediction == 1 else "Négatif"
        probability = self.classifier.predict_proba(text)[0][1] if prediction == 1 else  self.classifier.predict_proba(text)[0][0] 
        print(sentiment)
        print(probability)
        return sentiment, float("{:.2f}".format(probability * 100))
    
    # -- Make a prediction based on the user-entered data : either a unique string text or a list of text 
    def get_predictions(self,text):
        # -- multiple texts prediction
        if isinstance(text, list):
            print(f"Prediction of {len(text)} texts...")
            sentiments =[]
            probabilities =[]
            for item in text:
                sentiment, probability  = self.prediction(item)
                sentiments.append(sentiment)
                probabilities.append(probability)
            return sentiments, probabilities
        # -- unique text prediction
        if isinstance(text, str):
            sentiment, probability  = self.prediction(text)
            return sentiment, probability 
            

def get_model():
    if model_settings.MODEL_name=="Camembert_model":
        model = BertModel(model_settings.MODEL_DIR, model_settings.MODEL_filename)
    else: 
        model = Model(model_settings.MODEL_DIR, model_settings.MODEL_filename)
    return model
    

if __name__ == "__main__":
    #model = BertModel(model_settings.MODEL_DIR, model_settings.MODEL_filename)
    model = get_model()
    #raw_texts = ["J'aime ce restaurant, les plats sont délicieux", "C'était horrible, tout était mauvais."]
    raw_texts = [
        "C'était horrible, tout était mauvais."
        ,"Une cuisine d'Eric Prat délicieusement originale, subtile,raffinée servie avec une attention judicieuse, délicate et souriante font de chacun de nos repas un temps de bonheur gastronomique et d'humanité absolu. Merci et bravo!"
        ,"Nous avons été très déçu de la qualité du repas. On a pris une fondue bourguignonne et une fondue grenier à sel pas assez de viande et le les aiguillettes de canard étaient racis."
        ,"Lorsque nous sommes sur le secteur, nous y allons systématiquement. La cuisine, traditionnelle, est toujours excellente. Le cadre est chaleureux, et le personnel vraiment très sympathique"
        ]
    #raw_texts = "C'était horrible, tout était mauvais."
    #model.prediction(raw_texts)
    sentiment, probability =  model.get_predictions(raw_texts)
    print(sentiment, probability)

    
