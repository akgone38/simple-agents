# Chatbot Persistence with SQLite Checkpointer

This project demonstrates state persistence for a LangGraph chatbot using SQLite. The chatbot collects a user's name and interest across multiple steps, pauses execution, saves the session state to a SQLite database (`state.db`), and successfully resumes the session upon restart.

## Business Scenario
A sales chatbot gathers customer preferences across multiple interactions. It should be able to resume the session even after a shutdown or process restart without losing the customer's state.

## Implementation Details
The agent is designed using **LangGraph**:
1. **Nodes**:
   - `get_name`: Simulates getting the customer's name (sets name to `'Ravi'`).
   - `get_interest`: Simulates getting the customer's interest (sets interest to `'fitness'`).
   - `recommend`: Recommends a plan based on the customer's interest (sets recommendation to `'Recommended plan: Home Gym Starter Kit.'` for fitness).
2. **Persistence Layer**:
   - Uses `SqliteSaver` from `langgraph.checkpoint.sqlite`.
   - Compiles the StateGraph with the SQLite saver as the checkpointer.
   - Configures the compiler to stop/interrupt after the second node (`get_interest`) using `interrupt_after=["get_interest"]`.
3. **Execution Flow Control**:
   - On the first run, the script initializes a new SQLite checkpointer (`state.db`) and executes the graph. LangGraph interrupts the workflow after the first 2 nodes.
   - On the second run (rerun), the script detects `state.db`, restores the saved state, prints it, resumes execution of the third node (`recommend`), and outputs the final recommendation.

## Directory Structure
```text
chatbot_persistence/
├── .venv/               # Python Virtual Environment
├── README.md            # Project documentation (this file)
├── chatbot_persistence.py # Python script containing the graph and persistence logic
├── requirements.txt     # Python dependencies
└── state.db             # SQLite database created dynamically to store session states
```

## Setup Instructions

### 1. Set Up Virtual Environment & Dependencies
Initialize the virtual environment and install the required libraries:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Initial Session (Run 1)
Run the script to start the session, execute the first 2 nodes, save the checkpoint, and interrupt:
```bash
python3 chatbot_persistence.py
```

**Expected Output:**
```text
--- Run 1: Initializing Session & Running First 2 Nodes ---
State saved to database. Next scheduled node: ('recommend',)
Run 1 complete. Run the script again to resume from the saved state.
```

### 3. Resume the Session (Run 2)
Run the script again to load the saved state from the database, print the restored state, and finish the execution:
```bash
python3 chatbot_persistence.py
```

**Expected Output:**
```text
--- Run 2: Resuming Session from Database ---
Restored State: {'name': 'Ravi', 'interest': 'fitness'}

Final Output: 'Recommended plan: Home Gym Starter Kit.'
```
