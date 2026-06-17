import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Define the state schema
class AgentState(TypedDict):
    name: Optional[str]
    interest: Optional[str]
    recommendation: Optional[str]

# Define the node functions
def get_name(state: AgentState):
    return {"name": "Ravi"}

def get_interest(state: AgentState):
    return {"interest": "fitness"}

def recommend(state: AgentState):
    interest = state.get("interest")
    if interest == "fitness":
        rec = "Recommended plan: Home Gym Starter Kit."
    else:
        rec = "Recommended plan: Standard Kit."
    return {"recommendation": rec}

def main():
    db_path = "state.db"
    config = {"configurable": {"thread_id": "session_ravi_1"}}

    # Build the StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_name", get_name)
    workflow.add_node("get_interest", get_interest)
    workflow.add_node("recommend", recommend)
    
    # Set up flow edges
    workflow.add_edge(START, "get_name")
    workflow.add_edge("get_name", "get_interest")
    workflow.add_edge("get_interest", "recommend")
    workflow.add_edge("recommend", END)

    # Determine if this is Run 1 (initial) or Run 2 (resume)
    is_resume = os.path.exists(db_path)

    # Compile the graph with SQLite-based persistence
    with SqliteSaver.from_conn_string(db_path) as memory:
        # We interrupt immediately after get_interest (the second node)
        app = workflow.compile(checkpointer=memory, interrupt_after=["get_interest"])

        if not is_resume:
            print("--- Run 1: Initializing Session & Running First 2 Nodes ---")
            app.invoke({"name": "", "interest": "", "recommendation": ""}, config)
            print("State saved to database. Next scheduled node: ('recommend',)")
            print("Run 1 complete. Run the script again to resume from the saved state.")
        else:
            print("--- Run 2: Resuming Session from Database ---")
            
            # Load current state values
            state_info = app.get_state(config)
            name = state_info.values.get("name")
            interest = state_info.values.get("interest")
            print(f"Restored State: {{'name': '{name}', 'interest': '{interest}'}}")
            print()
            
            # Resume graph execution (input=None tells LangGraph to use checkpoint)
            final_state = app.invoke(None, config)
            recommendation = final_state.get("recommendation")
            print(f"Final Output: '{recommendation}'")
            
            # Clean up the DB for any future runs
            try:
                os.remove(db_path)
            except OSError:
                pass

if __name__ == "__main__":
    main()
