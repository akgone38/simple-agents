import os
import sys
import sqlite3
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# =====================================================================
# 1. INITIALIZATION & ENVIRONMENT SETUP
# =====================================================================

# Load environment variables from the local .env file.
# This makes HUGGINGFACEHUB_API_TOKEN available for authenticating LLM queries.
load_dotenv()

# Verify that the required third-party LangChain and Hugging Face modules can be imported.
try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as e:
    print(f"\n[ERROR] LangChain / HuggingFace imports failed: {e}")
    print("Please make sure you have activated the virtual environment and installed requirements.txt.")
    sys.exit(1)

# Initialize the HuggingFace LLM Endpoint.
# We point to Llama-3-8B-Instruct. We use a low temperature (0.1) for analytical stability.
try:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.1,
        provider="auto",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)
except Exception as e:
    print(f"\n[ERROR] Failed to initialize HuggingFace LLM: {e}")
    print("Please check your HUGGINGFACEHUB_API_TOKEN in .env.")
    sys.exit(1)

# Define the database path for persisting graph execution states
DB_PATH = "state.db"

# =====================================================================
# 2. STATE DEFINITION
# =====================================================================

# The AgentState defines the schema of the shared data passed between nodes.
class AgentState(TypedDict):
    expression: str   # The Python expression provided by the user
    approval: str     # The human approval decision: "yes" or "no"
    result: str       # The output of evaluating the expression or skipped status

# =====================================================================
# 3. GRAPH NODES (Core Execution Logic)
# =====================================================================

def parse_expression(state: AgentState):
    """
    Node 1: Analyzes the user's Python expression.
    Uses the HuggingFace LLM to predict the result and assess safety,
    then prints out the parsed expression.
    """
    expression = state.get("expression", "")
    print(f"\nExpression parsed: {expression}")
    
    # System instruction for instructing the LLM to inspect python code safety and predict outputs
    system_instruction = (
        "You are a Code Evaluation Agent. Analyze the user's Python expression.\n"
        "1. Predict the result of the expression.\n"
        "2. Identify any potential security or runtime risks (e.g., file operations, library imports, division by zero).\n"
        "Respond in a single sentence like: 'LLM Prediction: [predicted result]. Safety: [safe/unsafe - reason].'"
    )
    
    try:
        response = chat_model.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Expression: {expression}")
        ])
        print(f"Analysis: {response.content.strip()}")
    except Exception as e:
        print(f"[Warning] LLM analysis failed: {e}")
        
    return {"expression": expression}

def human_approve(state: AgentState):
    """
    Node 2: Halts the graph execution to request human-in-the-loop (HIL) approval.
    Calling `interrupt()` pauses graph execution, saves the checkpoint state,
    and returns control to the runner.
    When the runner resumes execution, this call returns the value supplied to Command(resume=value).
    """
    approval_response = interrupt(
        f"Do you approve execution of the expression '{state['expression']}'? (yes/no): "
    )
    # Normalize the string response (strip spaces and convert to lowercase)
    approval_str = str(approval_response).strip().lower()
    return {"approval": approval_str}

def execute_code(state: AgentState):
    """
    Node 3: Evaluates the expression using standard Python eval() function.
    Only executed if human approved the expression ("yes").
    """
    print("\nHuman approved execution.")
    expression = state.get("expression", "")
    try:
        # Evaluate python expression.
        # Standard evaluation catching errors to ensure program stability.
        res = eval(expression, {}, {})
        result_str = str(res)
    except Exception as e:
        result_str = f"Error during execution: {e}"
        
    return {"result": result_str}

def handle_reject(state: AgentState):
    """
    Node 4: Executed if human rejected the expression ("no").
    Updates state to indicate execution was skipped.
    """
    print("\nHuman rejected execution.")
    return {"result": "Execution skipped."}

# =====================================================================
# 4. CONDITIONAL EDGES / ROUTING
# =====================================================================

def route_approval(state: AgentState):
    """
    Routes the graph control flow depending on approval.
    - If "yes" -> execute_code node.
    - Else -> handle_reject node.
    """
    approval = state.get("approval", "").lower().strip()
    if approval == "yes":
        return "execute"
    else:
        return "reject"

# =====================================================================
# 5. GRAPH ASSEMBLY & COMPILATION
# =====================================================================

def build_evaluation_graph(checkpointer):
    """
    Assembles nodes, standard edges, and conditional routing edges into a StateGraph
    and compiles it using SQLite checkpoint persistence.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Register nodes
    workflow.add_node("parse_expression", parse_expression)
    workflow.add_node("human_approve", human_approve)
    workflow.add_node("execute_code", execute_code)
    workflow.add_node("handle_reject", handle_reject)
    
    # 2. Add sequential flow edges
    workflow.add_edge(START, "parse_expression")
    workflow.add_edge("parse_expression", "human_approve")
    
    # 3. Add conditional routing out of human_approve
    workflow.add_conditional_edges(
        "human_approve",
        route_approval,
        {
            "execute": "execute_code",
            "reject": "handle_reject"
        }
    )
    
    # 4. Connect terminal nodes to END
    workflow.add_edge("execute_code", END)
    workflow.add_edge("handle_reject", END)
    
    # 5. Compile with SQLite Checkpoint persistence
    return workflow.compile(checkpointer=checkpointer)

# =====================================================================
# 6. SIMULATION RUNNER (Execution Entrypoint)
# =====================================================================

def run_evaluation_flow(app, thread_id: str, expression: str, simulated_approval: str):
    """
    Executes a full evaluation session for a specific thread, input expression,
    and simulated approval.
    Shows the pause at the interrupt point, resumption with Command(resume=...),
    and final state values.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # --- PHASE 1: Start Execution ---
    print(f"\n--- Thread {thread_id}: Evaluating '{expression}' ---")
    
    # Run the graph. It will run parse_expression, enter human_approve,
    # hit the interrupt() call, save the checkpoint, and pause execution.
    app.invoke({"expression": expression, "approval": "", "result": ""}, config)
    
    # Check the current state of the thread to confirm it is paused at an interrupt
    state_snapshot = app.get_state(config)
    print(f"[Graph Status] Next scheduled node: {state_snapshot.next}")
    
    # Verify if the graph is currently waiting on an interrupt
    if state_snapshot.next and "human_approve" in state_snapshot.next:
        print(f"[Interrupt Triggered] Execution paused. Requesting approval...")
        print(f"Simulating human approval: {simulated_approval}")
        
        # --- PHASE 2: Resume Execution ---
        # Resume the graph by passing Command(resume=simulated_approval) to the invoke call.
        # This supplies the value returned by the interrupt() function inside human_approve.
        final_state = app.invoke(Command(resume=simulated_approval), config)
        
        # Print results of the execution
        print(f"\nResult: {final_state['result']}")
    else:
        print("[Error] Graph did not pause at the expected interrupt point.")

def main():
    # Remove existing database file if it exists to ensure a clean simulation run
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except OSError:
            pass

    # Compile the graph using SQLite persistence checkpointer
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        app = build_evaluation_graph(memory)
        
        # Case 1 — Approved
        run_evaluation_flow(
            app=app,
            thread_id="thread_approved",
            expression="5 * (3 + 2)",
            simulated_approval="yes"
        )
        
        # Case 2 — Rejected
        run_evaluation_flow(
            app=app,
            thread_id="thread_rejected",
            expression="open('file.txt', 'w')",
            simulated_approval="no"
        )
        
    # Cleanup database file after execution
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except OSError:
            pass

if __name__ == "__main__":
    main()
