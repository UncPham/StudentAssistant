# main.py

from fastapi import FastAPI, Body
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # BaseModel for structured data data models
from typing import List  # List type hint for type annotations
from langchain_community.tools.tavily_search import TavilySearchResults  # TavilySearchResults tool for handling search results from Tavily
import os  
from langgraph.prebuilt import create_react_agent  # Function to create a ReAct agent
from langchain_groq import ChatGroq  # ChatGroq class for interacting with LLMs
from dotenv import load_dotenv
# from backend import prompts
from datetime import datetime, timezone
from typing import List

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

tool_tavily = TavilySearchResults(max_results=5) 

tools = [tool_tavily]

app = FastAPI(debug=True)

SYSTEM_PROMPT = """You are an AI assistant

System time: {system_time}
"""

system_prompt = SYSTEM_PROMPT.format(
    system_time=datetime.now(tz=timezone.utc).isoformat()
)

origins = [
    "http://localhost:5173",
    # Add more origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestState(BaseModel):
    messages: List[str]

@app.post("/chat")
def chat_endpoint(request: RequestState = Body(...)):
    # Initialize the LLM with the selected model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    # Create a ReAct agent using the selected LLM and tools
    agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt)

    # Create the initial state for processing
    state = {"messages": request.messages}

    # Process the state using the agent
    result = agent.invoke(state)  # Invoke the agent (can be async or sync based on implementation)

    # Return the result as the response
    return result

# app.include_router(user.router)
# app.include_router(auth.router, prefix="/auth")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)