import os
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")