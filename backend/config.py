import os
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MONGODB_CONNECTION_URI=os.getenv("MONGODB_CONNECTION_URI")
WEAVIATE_API_KEY=os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL=os.getenv("WEAVIATE_URL")

CACHE_DIR = "models_cache"


