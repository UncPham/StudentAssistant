# main.py
from fastapi import FastAPI, Body, Request, APIRouter, HTTPException
from starlette.responses import RedirectResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport
from starlette.config import Config
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
import os  
  # Function to create a ReAct agent

from dotenv import load_dotenv
from config import CLIENT_ID, CLIENT_SECRET, SECRET_KEY, GROQ_API_KEY, GOOGLE_API_KEY, MONGODB_CONNECTION_URI, WEAVIATE_API_KEY, WEAVIATE_URL
# from backend import prompts
from authlib.integrations.starlette_client import OAuth, OAuthError
# from fastapi.staticfiles import StaticFiles
from httpx_oauth.clients.google import GoogleOAuth2

from fastapi_users.authentication import CookieTransport
from fastapi_users.authentication import JWTStrategy

from api.api import router 

import google.oauth2.credentials
# import google_auth_oauthlib.flow
from pymongo import MongoClient

import weaviate
from weaviate.classes.init import Auth

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

cookie_transport = CookieTransport(cookie_max_age=3600)

load_dotenv()

app = FastAPI()

config_data = {'GOOGLE_CLIENT_ID': CLIENT_ID, 'GOOGLE_CLIENT_SECRET': CLIENT_SECRET}


origins = [
    "http://localhost:5173",
]

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)

starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/o/oauth2/auth',
    client_kwargs={'scope': 'openid email profile'},
)

@app.on_event("startup")
def startup_db_client():
    # app.mongodb_client = MongoClient(MONGODB_CONNECTION_URI)
    # app.database = app.mongodb_client["StudentAssistant"]
    # print("Connected to the MongoDB database!")

    # Kết nối Weaviate
    app.weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )
    print("Connected to Weaviate!")
    

@app.on_event("shutdown")
def shutdown_db_client():
    # app.mongodb_client.close()
    # Đóng kết nối Weaviate
    app.weaviate_client = None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)