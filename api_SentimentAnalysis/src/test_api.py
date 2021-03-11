
import sys
import os 
import json
from fastapi.testclient import TestClient
#sys.path.insert(0,'../')
from main import app
from Model_api import *


# pour eviter erreur _bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed (due à pytorch) => downgrade numpy de 1.16 à 1.15.4
# Refs : https://towardsdatascience.com/from-scripts-to-prediction-api-2372c95fb7c7


# -- Create a Test client 
client = TestClient(app)

FAKE_SECRET_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

def test_welcome():
    response = client.get("/welcome")
    assert response.status_code == 200
    assert response.json() == {"Message": "Bonjour, ceci est la beta d'un algorithm d'analyse de sentiment utilisant FastAPI comme framework API","status Code": 200} 


def test_bad_token():
    response = client.post("/token/", headers={"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Token header"}


def test_good_token():
    response = client.post("/token/", headers={"token": FAKE_SECRET_TOKEN})
    assert response.status_code == 200
    assert response.json() == {"token" : FAKE_SECRET_TOKEN}


def test_model_ML():
    settings = SettingsModel()
    assert settings.MODEL_name ==  "LemmaTokenizer_TFIDFbigram_LinearSVM"

def test_model_BERT():
    settings = SettingsModel()
    assert settings.MODEL_name ==  "Camembert_model"

def test_single_api_predict():
    data = {"text":"C'était horrible, tout était mauvais."}
    expected_response = {
        "sentiment": "Négatif",
        "probability": 99.95,
    }
    # Test client uses "query_string" instead of "params"
    response = client.post('/predict/', json= data, headers={"token":FAKE_SECRET_TOKEN})
    # Check that we got "200 OK" back.
    assert response.status_code == 200
    response_json = response.json()
    # output shape and key check
    assert all(
        [key in response_json.keys() for key in ["sentiment", "probability"]]
    )
    assert len(response_json) == 2 # 2 outputs are expected
    # ouput type check
    assert isinstance(response_json["sentiment"], str)
    assert isinstance(response_json["probability"], float)
    # expected responde check 
    assert response_json== expected_response

    
def test_api_predict_ML():
    data_file = os.path.join('test_files','test_text_v1.json')
    # Load all the test cases
    with open(data_file) as f:
        test_data = json.load(f)
    for data in test_data:
        print(data)
        expected_response = data['expected_response']
        # Test client 
        response = client.post('/predict/', json= data, headers={"token":FAKE_SECRET_TOKEN})
        # Check that we got the correct status code back.
        assert response.status_code == 200
        # response.data returns a byte array, convert to a dict.
        assert response.json() == expected_response


    
def test_api_predict_BERT():
    data_file = os.path.join('test_files','test_text_v2.json')
    # Load all the test cases
    with open(data_file) as f:
        test_data = json.load(f)
    for data in test_data:
        print(data)
        expected_response = data['expected_response']
        # Test client 
        response = client.post('/predict/', json= data, headers={"token":FAKE_SECRET_TOKEN})
        # Check that we got the correct status code back.
        assert response.status_code == 200
        # response.data returns a byte array, convert to a dict.
        assert response.json() == expected_response












    

