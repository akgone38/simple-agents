import os
import sys
import warnings
warnings.filterwarnings("ignore")
from typing import TypedDict, Annotated, Sequence
import operator
import requests

from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_core.tools import tool
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import create_react_agent
except ImportError as e:
    print(f"\n[ERROR] Missing packages: {e}")
    sys.exit(1)

# =====================================================================
# 1. INITIALIZE NATIVE GROQ ENGINE
# =====================================================================

def get_groq_model():
    """Initializes the native Groq model with built-in tool calling support."""
    token = os.getenv("GROQ_API_KEY")
    if not token:
        print("[Warning] GROQ_API_KEY is missing from your .env file.")
        return None
        
    try:
        # Llama-3.1 natively supports strict tool calling via Groq's API
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=token
        )
    except Exception as e:
        print(f"[Warning] Failed to initialize Groq model: {e}")
        return None

# =====================================================================
# 2. AGENT TOOLS
# =====================================================================
@tool
def get_weather(city: str) -> str:
    """Get the current live temperature for a given city."""
    print(f"\n[Tool Execution] Fetching live weather for: {city}...")
    try:
        # Step 1: Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        
        if "results" not in geo_res:
            error_msg = f"Could not find coordinates for {city}."
            print(f"[Tool Result] {error_msg}")
            return error_msg
            
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        
        # Step 2: Fetch Weather (Using the updated Open-Meteo API format)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
        weather_res = requests.get(weather_url).json()
        
        # Step 3: Extract the temperature
        if "current" not in weather_res:
            error_msg = f"Live weather data format changed or is unavailable for {city}."
            print(f"[Tool Result] {error_msg}")
            return error_msg
            
        temp = weather_res["current"]["temperature_2m"]
        
        success_msg = f"The current live temperature in {city} is {temp}°C."
        print(f"[Tool Result] {success_msg}") # <--- This will show you exactly what the AI sees!
        return success_msg
        
    except Exception as e:
        error_msg = f"Error fetching weather: {str(e)}"
        print(f"[Tool Result] {error_msg}")
        return error_msg
# =====================================================================
# 3. LANGGRAPH WORKFLOW ORCHESTRATION
# =====================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def create_supervisor(agent_runnable):
    workflow = StateGraph(AgentState)
    
    def run_agent(state: AgentState):
        return {"messages": agent_runnable.invoke(state)["messages"]}
        
    def run_supervisor(state: AgentState):
        messages = state.get("messages", [])
        if messages:
            print(f"\nFinal Agent Response: “{messages[-1].content}”")
        return {}
        
    workflow.add_node("agent", run_agent)
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "supervisor")
    workflow.add_edge("supervisor", END)
    
    return workflow.compile()

# =====================================================================
# 4. EXECUTION
# =====================================================================

def main():
    engine = get_groq_model()
    if not engine:
        sys.exit(1)
        
    tools = [get_weather]
    react_agent = create_react_agent(engine, tools=tools)
    supervised_agent = create_supervisor(react_agent)
    
    test_query = "What is the current weather in Austin?"
    print(f"User Query: {test_query}")
    supervised_agent.invoke({"messages": [HumanMessage(content=test_query)]})

if __name__ == "__main__":
    main()