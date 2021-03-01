#%% api.py

# -- Packages 

import os 

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException,  Security
from fastapi.security.api_key import APIKeyQuery, APIKeyHeader, APIKey
#from enum import Enum
from pydantic import BaseModel,BaseSettings 
import logging
import time
from datetime import datetime
from typing import Dict
import sys
#sys.path.append('.')
from Model_api import Model, get_model
from log_api import Log
#from test_api import TestClient


# -- FastAPI application instance
# - The @app.get("/") tells FastAPI that the function right below is in charge of handling requests that go to /Welcome using a get operation
# - get -> for reading data ; the server returns something
app = FastAPI()


# -- log Settings 
class SettingsLog(BaseSettings):
    LOG_DIR: str = os.path.join(os.getcwd(),'..', 'logs')
    LOG_FILE: str = 'SentimentAnalysis_fastAPI'
    LOGGER_NAME: str = 'SentimentAnalysis_fastAPI'

settings = SettingsLog()
api_log = Log(settings.LOG_FILE, log_dir= settings.LOG_DIR).get_logger(settings.LOGGER_NAME, level=logging.INFO)


# -- token authentification
FAKE_SECRET_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

@app.post("/token/")
async def token_auth(token: str = Header(...)):
    if token != FAKE_SECRET_TOKEN:
        api_log.error('Connexion failed with invalid Token header %s',token)
        raise HTTPException(status_code=400, detail="Invalid Token header")
    else:
        api_log.info('Connexion with valid Token header%s',token) 
    return token

# -- Request from a client 
class SentimentRequest(BaseModel):
    text: str

# -- Response from the server 
class SentimentResponse(BaseModel):
    sentiment: str
    probability: float

@app.get("/welcome")
async def welcomeMessage():
    api_log.info('USER Connexion to Welcome')
    return {"Message": "Bonjour, ceci est la beta d'un algorithm d'analyse de sentiment utilisant FastAPI comme framework API","status Code": 200} 

@app.post("/predict/", response_model=SentimentResponse)
async def predict(request: SentimentRequest, model: Model = Depends(get_model), token: str = Depends(token_auth)):
    print("Predicting")
    sentiment, probability  = model.predict([request.text])
    api_log.info(f"token: {token}")
    api_log.info(f"Text request: {request.text} | Prediction: {sentiment} | Probability: {probability}")
    return SentimentResponse(
        sentiment=sentiment,
        probability = probability 
    )



@app.put("/Logs/{query_date}") # log_instance: Log = Depends(Log)
async def read_log(query_date: str):
    api_log.info('USER Connexion to Welcome')
    query_log = Log(settings.LOG_FILE, query_date).read_logs()
    print(query_log)
    return  {"Log Date": query_date, "Messages": query_log}


if __name__ == "__main__":
    # --- FastAPI --- #
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)




# X_API_TOKEN = APIKeyHeader(name='X-API-token')

# def check_authentication_header(x_api_key: str = Depends(X_API_TOKEN)):
#     if x_api_key == FAKE_SECRET_TOKEN :
#         api_log.info('Connexion with valid Token header%s',token) 
#         return {"Token": FAKE_SECRET_TOKEN}
#     # else raise 401
#     else: 
#         api_log.error('Connexion failed with invalid Token header %s',token)
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid API Token",
#         )

# @app.get("/result/", response_model=List[Result])
# def result(user: User = Depends(check_authentication_header)):
#     """ return a list of test results """
#     print('user', user)

 