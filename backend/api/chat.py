from fastapi import APIRouter, Request, Body
import os
import sys
from langgraph.prebuilt import create_react_agent 
from langchain_groq import ChatGroq  # ChatGroq class for interacting with LLMs
from langchain_community.tools.tavily_search import TavilySearchResults  # TavilySearchResults tool for handling search results from Tavily
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel  # BaseModel for structured data data models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.config import GROQ_API_KEY, HUGGINGFACE_API_KEY
from backend.prompts import SYSTEM_PROMPT
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tool_tavily = TavilySearchResults(max_results=5) 
tools = [tool_tavily]

system_prompt = SYSTEM_PROMPT.format(
    system_time=datetime.now(tz=timezone.utc).isoformat()
)

class RequestState(BaseModel):
    messages: List[str]

@router.post("/chat")
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