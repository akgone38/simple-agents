import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# 1. Load Environment Variables
load_dotenv()

# 2. Define the Expanded State Schema
class BotState(TypedDict):
    query: str
    type: str  # "faq", "complaint", "technical_support", or "escalate"
    response: str

# 3. Initialize the Chat Model Pipeline
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.3, # Slightly higher temperature for richer language generation
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
chat_model = ChatHuggingFace(llm=llm_endpoint)

# 4. Node 1: Multi-Intent Classification Node
def classify_query(state: BotState) -> dict:
    """Categorizes the user's message into one of four distinct intents."""
    query = state["query"]
    
    messages = [
        SystemMessage(content=(
            "You are an advanced routing bot for customer operations. "
            "Classify the user message into exactly one of these lowercase words: "
            "'faq', 'complaint', 'technical_support', or 'escalate'.\n\n"
            "Classification Rules:\n"
            "- 'faq': Simple, general business inquiries (hours, returns, tracking, shipping rules).\n"
            "- 'complaint': Expressing frustration, bugs, delays, anger, or demanding a refund.\n"
            "- 'technical_support': Specific issues regarding hardware, software setup, code, errors, or configuration.\n"
            "- 'escalate': High-value corporate requests, legal threats, or explicitly asking for a human manager.\n\n"
            "Respond with ONLY the single word category name. No extra text, punctuation, or preamble."
        )),
        HumanMessage(content=f"Message: {query}")
    ]
    
    response = chat_model.invoke(messages)
    category = response.content.strip().lower()
    
    # Strict validation mapping layer
    valid_intents = {"faq", "complaint", "technical_support", "escalate"}
    if category not in valid_intents:
        # Check substring fallbacks if the LLM added filler words
        for intent in valid_intents:
            if intent in category:
                category = intent
                break
        else:
            category = "escalate"  # Default safety fallback
        
    print(f"[Node: classify_query] -> Intent Classified as: '{category}'")
    return {"type": category}

# 5. Node 2: Dynamic FAQ Node
def faq_node(state: BotState) -> dict:
    """Generates a dynamic answer based on standard store parameters."""
    query = state["query"]
    messages = [
        SystemMessage(content=(
            "You are a helpful customer support bot answering FAQs. Answer the user's question accurately based on these facts:\n"
            "- Shipping: 3-5 business days domestically, 10-14 days internationally.\n"
            "- Returns: 30-day window, items must be unopened.\n"
            "- Store Hours: 9 AM - 6 PM EST, Mon-Fri.\n"
            "Keep the reply professional, polite, and concise (under 2 sentences)."
        )),
        HumanMessage(content=query)
    ]
    response = chat_model.invoke(messages)
    return {"response": response.content.strip()}

# 6. Node 3: Dynamic Complaint Log & De-escalation Node
def complaint_node(state: BotState) -> dict:
    """Logs the issue and generates an empathetic, professional apology."""
    query = state["query"]
    messages = [
        SystemMessage(content=(
            "You are an empathetic customer recovery specialist. "
            "Acknowledge the user's frustration, apologize sincerely, and inform them that an incident ticket "
            "has been logged with our QA team for immediate investigation. Keep it reassuring and short."
        )),
        HumanMessage(content=query)
    ]
    response = chat_model.invoke(messages)
    return {"response": response.content.strip()}

# 7. Node 4: Dynamic Technical Troubleshooting Node
def technical_support_node(state: BotState) -> dict:
    """Provides targeted, structured, step-by-step guidance for technical issues."""
    query = state["query"]
    messages = [
        SystemMessage(content=(
            "You are an expert IT Tier-1 technical support engineer. Provide the user with 2 clear, bulleted troubleshooting "
            "steps to address their technical problem based on their message. Keep the tone helpful and technical."
        )),
        HumanMessage(content=query)
    ]
    response = chat_model.invoke(messages)
    return {"response": response.content.strip()}

# 8. Node 5: Human Escalation Node
def escalate_node(state: BotState) -> dict:
    """Prepares standard handoff protocols."""
    return {"response": "This request requires executive routing or direct account management. I am transferring this interaction immediately to a senior agent. Live chat queue time is currently < 2 minutes."}

# 9. Define Conditional Routing Logic
def route_based_on_type(state: BotState) -> str:
    """Maps state['type'] exactly to the string name of the next Graph node."""
    mapping = {
        "faq": "faq_node",
        "complaint": "complaint_node",
        "technical_support": "technical_support_node",
        "escalate": "escalate_node"
    }
    return mapping.get(state["type"], "escalate_node")

# 10. Build and Compile the Workflow StateGraph
workflow = StateGraph(BotState)

# Register the nodes
workflow.add_node("classify_query", classify_query)
workflow.add_node("faq_node", faq_node)
workflow.add_node("complaint_node", complaint_node)
workflow.add_node("technical_support_node", technical_support_node)
workflow.add_node("escalate_node", escalate_node)

# Set up standard edge flow
workflow.add_edge(START, "classify_query")

# Set up conditional branching
workflow.add_conditional_edges(
    "classify_query",
    route_based_on_type,
    {
        "faq_node": "faq_node",
        "complaint_node": "complaint_node",
        "technical_support_node": "technical_support_node",
        "escalate_node": "escalate_node"
    }
)

# Close all endpoints to standard termination
workflow.add_edge("faq_node", END)
workflow.add_edge("complaint_node", END)
workflow.add_edge("technical_support_node", END)
workflow.add_edge("escalate_node", END)

app = workflow.compile()

# 11. Multivariant Verification Run
if __name__ == "__main__":
    test_cases = [
        {"name": "Complaint Test", "query": "I ordered overnight shipping and it's been 4 days! I want my money back."},
        {"name": "FAQ Test", "query": "Can you tell me what time your support line closes on Fridays?"},
        {"name": "Technical Support Test", "query": "My application is crashing with a '401 unauthorized connection' token error whenever I push data."},
        {"name": "Escalation Test", "query": "Get me a human manager right away. Your bot is frustrating me and I will cancel our corporate license."}
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        result = app.invoke({"query": case["query"]})
        print(f"Final Output Response:\n{result['response']}\n")