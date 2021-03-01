
import sys
#sys.path.append('../')
from fastapi.testclient import TestClient
from api import app
import json


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


# @app.post("/token/")
# async def token_auth(token: str = Header(...)):
#     if token != FAKE_SECRET_TOKEN:
#         api_log.error('Connexion failed with invalid Token header %s',token)
#         raise HTTPException(status_code=400, detail="Invalid Token header")
#     else:
#         api_log.info('Connexion with valid Token header%s',token) 
#     return token


# def test_create_item_bad_token():
#     response = client.post(
#         "/items/",
#         headers={"X-Token": "hailhydra"},
#         json={"id": "bazz", "title": "Bazz", "description": "Drop the bazz"},
#     )
#     assert response.status_code == 400
#     assert response.json() == {"detail": "Invalid X-Token header"}

    
# @app.post("/items/", response_model=Item)
# async def create_item(item: Item, x_token: str = Header(...)):
#     if x_token != fake_secret_token:
#         raise HTTPException(status_code=400, detail="Invalid X-Token header")
#     if item.id in fake_db:
#         raise HTTPException(status_code=400, detail="Item already exists")
#     fake_db[item.id] = item
#     return item


# def test_model():
#     settings = SettingsModel()
#     assert settings.MODEL_name == 'ML_model'


# def test_single_api_predict():
#     data = {"text":"C'était horrible, tout était mauvais."}
#     expected_response = {
#         "sentiment": "Négatif",
#         "probability": 99.95,
#     }

#     # Test client uses "query_string" instead of "params"
#     response = client.post('/predict/', headers={"token":FAKE_SECRET_TOKEN}, request_text = data}
#     # Check that we got "200 OK" back.
#     assert response.status_code == 200
#     # response.data 
#     assert response.json() == expected_response



    

